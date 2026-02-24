# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    T_leftCam_device: SE3, T_rightCam_device: SE3
) -> tuple[SO3, SO3]:
    """
    Compute the rotations necessary to scanline rectify two cameras.

    Scanline rectification aligns the epipolar lines of both cameras to be
    horizontal, making stereo matching more efficient (1D search instead of 2D).

    Args:
        T_leftCam_device: Transform from device frame to left camera frame
        T_rightCam_device: Transform from device frame to right camera frame

    Returns:
        Tuple of (R_left_rect, R_right_rect) -- rotations that map points from
        the rectified frame to the original left/right camera frames.
    """
    # Compute R to L transform: R -> device -> L
    T_leftCam_rightCam = T_leftCam_device @ T_rightCam_device.inverse()

    R_leftCam_rightCam = T_leftCam_rightCam.rotation()
    baseline_in_leftCam = T_leftCam_rightCam.translation()

    # Up vector (in left camera frame, Y points down in image)
    up_in_leftCam = np.array([0, -1, 0])

    # Hypothetical fwd vector for each camera, perpendicular to baseline (in left FoR)
    fwd_in_leftCam = normalize(np.cross(up_in_leftCam, normalize(baseline_in_leftCam)))
    if np.linalg.norm(fwd_in_leftCam) < 1e-6:
        # Singular case when up_in_leftCam and baseline_in_leftCam are parallel
        fwd_in_leftCam = np.array([0, 0, 1])

    # Define new basis (in left FoR)
    x_axis_rect = normalize(baseline_in_leftCam)
    z_axis_rect = fwd_in_leftCam
    y_axis_rect = normalize(np.cross(z_axis_rect, x_axis_rect))

    # New orientation for both left and right cameras (expressed relative to original left)
    R_left_rect_matrix = np.vstack([x_axis_rect, y_axis_rect, z_axis_rect]).transpose()
    R_left_rect = SO3.from_matrix(R_left_rect_matrix)
    R_right_rect = R_leftCam_rightCam.inverse() @ R_left_rect

    return R_left_rect, R_right_rect


def fisheye_to_linear_calib(
    calib: calibration.CameraCalibration,
    focal_scale: float = 1.25,
    output_width: int = None,
    output_height: int = None,
    use_original_pp: bool = True,
) -> calibration.CameraCalibration:
    """
    Create a linear (pinhole) camera calibration based on fisheye calibration.

    The linear model is useful for stereo matching as it removes lens distortion
    and provides a simpler projection model.

    Args:
        calib: Original fisheye camera calibration
        focal_scale: Scale factor for focal length (default 1.25 provides good FOV)
        output_width: Width of the output rectified image (None = source image width)
        output_height: Height of the output rectified image (None = source image height)
        use_original_pp: If True, use the fisheye camera's original cx/cy as the
            principal point. If False, center the principal point at (w/2, h/2).

    Returns:
        New CameraCalibration with LINEAR model type
    """
    img_size = calib.get_image_size()
    if output_width is None:
        output_width = img_size[0]
    if output_height is None:
        output_height = img_size[1]

    params = calib.get_projection_params()
    linear_focal = params[0] * focal_scale

    if use_original_pp:
        # Use the fisheye camera's original principal point.
        # Fisheye624 param ordering: [f, cx, cy, ...]
        cx = float(params[1])
        cy = float(params[2])
    else:
        cx = output_width / 2.0
        cy = output_height / 2.0

    # Create linear intrinsics: [fx, fy, cx, cy]
    linear_params = np.array([linear_focal, linear_focal, cx, cy])

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


def compute_stereo_baseline(T_leftCam_device: SE3, T_rightCam_device: SE3) -> float:
    """
    Compute the baseline distance between stereo cameras in meters.

    Args:
        T_leftCam_device: Transform from device frame to left camera frame
        T_rightCam_device: Transform from device frame to right camera frame

    Returns:
        Baseline distance in meters
    """
    T_leftCam_rightCam = T_leftCam_device @ T_rightCam_device.inverse()
    return float(np.linalg.norm(T_leftCam_rightCam.translation()))


def rectify_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_calib: calibration.CameraCalibration,
    right_calib: calibration.CameraCalibration,
    left_linear: calibration.CameraCalibration,
    right_linear: calibration.CameraCalibration,
    R_left_rect: SO3,
    R_right_rect: SO3,
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
        R_left_rect: Rectification rotation for left camera
        R_right_rect: Rectification rotation for right camera
        interpolation: Interpolation method for resampling

    Returns:
        Tuple of (left_rectified, right_rectified) images
    """
    left_rectified = calibration.distort_by_calibration_and_apply_rotation(
        left_image, left_linear, left_calib, R_left_rect, interpolation
    )

    right_rectified = calibration.distort_by_calibration_and_apply_rotation(
        right_image, right_linear, right_calib, R_right_rect, interpolation
    )

    return left_rectified, right_rectified


def disparity_to_depth(
    disparity: np.ndarray,
    baseline: float,
    focal_length: float,
) -> np.ndarray:
    """
    Convert disparity map to depth map using stereo geometry.

    The relationship is: depth = baseline * focal_length / (disparity + epsilon)

    Args:
        disparity: Disparity map in pixels (H, W)
        baseline: Baseline distance between cameras in meters
        focal_length: Focal length in pixels

    Returns:
        Depth map in meters (H, W), float32
    """
    depth = (baseline * focal_length) / (disparity + 1e-6)
    return depth.astype(np.float32)


def compute_T_world_rectCam(
    T_world_device: SE3, T_leftCam_device: SE3, R_left_rect: SO3
) -> SE3:
    """
    Compute the world pose of the rectified left camera.

    Matches the pipeline:
        T_world_cam = T_world_device * T_device_cam
        T_left_rect = SE3(R_left_rect, 0)
        T_world_rectCam = T_world_cam * T_left_rect

    where R_left_rect maps points from the rectified camera space to the
    original left camera space.

    Args:
        T_world_device: Pose of the device in world frame
        T_leftCam_device: Transform from device frame to left camera frame
        R_left_rect: Rotation mapping points from rectified frame to original
            left camera frame

    Returns:
        SE3 pose of the rectified camera in world frame
    """
    T_world_cam = T_world_device @ T_leftCam_device.inverse()

    # Convert SO3 rotation to SE3 with zero translation
    quat = R_left_rect.to_quat()[0]  # [w, x, y, z]
    T_left_rect = SE3.from_quat_and_translation(
        quat[0], quat[1:4], np.array([0.0, 0.0, 0.0])
    )

    T_world_rectCam = T_world_cam @ T_left_rect
    return T_world_rectCam
