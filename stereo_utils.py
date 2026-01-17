# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Stereo Depth Estimation Utilities for Aria Gen2

This module provides utility functions for stereo rectification and depth
estimation from Aria Gen2 front-facing SLAM cameras.

Key functions:
- create_scanline_rectified_cameras: Compute rectification rotations
- fisheye_to_linear_calib: Convert fisheye to linear camera model
- rectify_stereo_pair: Apply rectification to image pair
- compute_stereo_baseline: Get baseline distance between cameras
- disparity_to_depth: Convert disparity map to depth map
"""

import numpy as np
from projectaria_tools.core import calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sophus import SE3, SO3


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def create_scanline_rectified_cameras(
    T_l_cam_device: SE3, T_r_cam_device: SE3
) -> tuple[SO3, SO3]:
    """
    Compute the rotations necessary to scanline rectify two cameras.

    Scanline rectification aligns the epipolar lines of both cameras to be
    horizontal, making stereo matching more efficient (1D search instead of 2D).

    Args:
        T_l_cam_device: Transform from device frame to left camera frame
        T_r_cam_device: Transform from device frame to right camera frame

    Returns:
        Tuple of (Rl_n, Rr_n) - rotation corrections for left and right cameras
        that align their epipolar lines horizontally.
    """
    # Compute R to L transform: R -> device -> L
    T_lr = T_l_cam_device @ T_r_cam_device.inverse()

    R_lr = T_lr.rotation()
    r_l = T_lr.translation()

    # Up vector (in left camera frame, Y points down in image)
    lup_l = np.array([0, -1, 0])

    # Hypothetical fwd vector for each camera, perpendicular to baseline (in left FoR)
    lfwd_l = normalize(np.cross(lup_l, normalize(r_l)))
    if np.linalg.norm(lfwd_l) < 1e-6:
        # Singular case when lup_l and r_l are parallel
        lfwd_l = np.array([0, 0, 1])

    # Define new basis (in left FoR)
    nx_l = normalize(r_l)
    nz_l = lfwd_l
    ny_l = normalize(np.cross(nz_l, nx_l))

    # New orientation for both left and right cameras (expressed relative to original left)
    Rl_n_mat = np.vstack([nx_l, ny_l, nz_l]).transpose()
    Rl_n = SO3.from_matrix(Rl_n_mat)
    Rr_n = R_lr.inverse() @ Rl_n

    return Rl_n, Rr_n


def fisheye_to_linear_calib(
    calib: calibration.CameraCalibration,
    focal_scale: float = 1.25,
    output_width: int = 512,
    output_height: int = 512,
) -> calibration.CameraCalibration:
    """
    Create a linear (pinhole) camera calibration based on fisheye calibration.

    The linear model is useful for stereo matching as it removes lens distortion
    and provides a simpler projection model.

    Args:
        calib: Original fisheye camera calibration
        focal_scale: Scale factor for focal length (default 1.25 provides good FOV)
        output_width: Width of the output rectified image
        output_height: Height of the output rectified image

    Returns:
        New CameraCalibration with LINEAR model type
    """
    # Scale the focal length from the fisheye model
    linear_focal = calib.get_projection_params()[0] * focal_scale

    # Create linear intrinsics: [fx, fy, cx, cy]
    linear_params = np.array(
        [
            linear_focal,
            linear_focal,
            output_width / 2.0,  # Principal point at image center
            output_height / 2.0,
        ]
    )

    linear_calib = calibration.CameraCalibration(
        calib.get_label() + "-linear",
        calibration.CameraModelType.LINEAR,
        linear_params,
        SE3(),  # Identity transform for device-to-camera (applied separately)
        output_width,
        output_height,
        None,
        calib.get_max_solid_angle(),
        calib.get_serial_number(),
    )

    return linear_calib


def compute_stereo_baseline(T_l_cam_device: SE3, T_r_cam_device: SE3) -> float:
    """
    Compute the baseline distance between stereo cameras in meters.

    Args:
        T_l_cam_device: Transform from device frame to left camera frame
        T_r_cam_device: Transform from device frame to right camera frame

    Returns:
        Baseline distance in meters
    """
    # Get camera positions in device frame
    T_device_l_cam = T_l_cam_device.inverse()
    T_device_r_cam = T_r_cam_device.inverse()

    # Camera origins in device frame
    left_origin = T_device_l_cam.translation()
    right_origin = T_device_r_cam.translation()

    # Baseline is the distance between camera origins
    baseline = np.linalg.norm(right_origin - left_origin)

    return float(baseline)


def rectify_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_calib: calibration.CameraCalibration,
    right_calib: calibration.CameraCalibration,
    left_linear: calibration.CameraCalibration,
    right_linear: calibration.CameraCalibration,
    Rl_n: SO3,
    Rr_n: SO3,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rectify a stereo image pair for scanline-aligned stereo matching.

    This function applies distortion correction and rotation to align
    the epipolar lines horizontally in both images.

    Args:
        left_image: Left camera image (H, W) or (H, W, C)
        right_image: Right camera image (H, W) or (H, W, C)
        left_calib: Original left camera calibration (fisheye)
        right_calib: Original right camera calibration (fisheye)
        left_linear: Target left linear camera calibration
        right_linear: Target right linear camera calibration
        Rl_n: Rectification rotation for left camera
        Rr_n: Rectification rotation for right camera
        interpolation: Interpolation method for resampling

    Returns:
        Tuple of (left_rectified, right_rectified) images
    """
    left_rectified = calibration.distort_by_calibration_and_apply_rotation(
        left_image, left_linear, left_calib, Rl_n, interpolation
    )

    right_rectified = calibration.distort_by_calibration_and_apply_rotation(
        right_image, right_linear, right_calib, Rr_n, interpolation
    )

    return left_rectified, right_rectified


def disparity_to_depth(
    disparity: np.ndarray,
    baseline: float,
    focal_length: float,
    min_disparity: float = 1.0,
    max_depth: float = 20.0,
) -> np.ndarray:
    """
    Convert disparity map to depth map using stereo geometry.

    The relationship is: depth = baseline * focal_length / disparity

    Args:
        disparity: Disparity map in pixels (H, W)
        baseline: Baseline distance between cameras in meters
        focal_length: Focal length in pixels
        min_disparity: Minimum valid disparity (to avoid division issues)
        max_depth: Maximum depth value to clip to (meters)

    Returns:
        Depth map in meters (H, W), with invalid pixels set to 0
    """
    # Create depth array
    depth = np.zeros_like(disparity, dtype=np.float32)

    # Find valid disparity values
    valid_mask = disparity > min_disparity

    # Convert disparity to depth: Z = baseline * focal / disparity
    depth[valid_mask] = (baseline * focal_length) / disparity[valid_mask]

    # Clip to reasonable range
    depth = np.clip(depth, 0, max_depth)

    # Set invalid regions to 0
    depth[~valid_mask] = 0

    return depth


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud in camera coordinates.

    Args:
        depth: Depth map in meters (H, W)
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        Point cloud as (N, 3) array of [X, Y, Z] points in camera frame
    """
    h, w = depth.shape

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Valid depth mask
    valid = (depth > min_depth) & (depth < max_depth)

    # Get valid depth values
    z = depth[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Backproject to 3D
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    # Stack into point cloud
    points = np.stack([x, y, z], axis=1)

    return points


def get_rectified_camera_transform(T_cam_device: SE3, R_n: SO3) -> SE3:
    """
    Compute the transform from device frame to rectified camera frame.

    Args:
        T_cam_device: Original transform from device to camera
        R_n: Rectification rotation applied to the camera

    Returns:
        Transform from device frame to rectified camera frame
    """
    # Convert SO3 rotation to SE3 (with zero translation)
    quat = R_n.to_quat()[0]  # Get quaternion as [w, x, y, z]
    T_n = SE3.from_quat_and_translation(quat[0], quat[1:4], np.array([0.0, 0.0, 0.0]))

    # Combined transform: device -> camera -> rectified
    # R_n transforms from original camera to rectified camera
    # So T_rect_device = R_n @ T_cam_device (apply R_n on the output side)
    T_rect_device = T_n @ T_cam_device

    return T_rect_device
