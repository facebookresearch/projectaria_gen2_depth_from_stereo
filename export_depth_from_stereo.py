#!/usr/bin/env python3
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

"""Export depth from stereo using Foundation Stereo."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

# Project Aria Tools
from projectaria_tools.core import data_provider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

# Foundation Stereo (git submodule at ./FoundationStereo)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOUNDATION_STEREO_PATH = os.path.join(SCRIPT_DIR, "FoundationStereo")

if not os.path.isfile(
    os.path.join(FOUNDATION_STEREO_PATH, "core", "foundation_stereo.py")
):
    raise FileNotFoundError(
        f"FoundationStereo not found at {FOUNDATION_STEREO_PATH}\n"
        "Initialize the git submodule:\n"
        "  git submodule update --init\n"
        "Or clone manually:\n"
        "  git clone https://github.com/NVlabs/FoundationStereo.git FoundationStereo"
    )

sys.path.insert(0, FOUNDATION_STEREO_PATH)

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from omegaconf import OmegaConf

# Local utilities
from stereo_utils import (
    compute_T_world_rectCam,
    create_scanline_rectified_cameras,
    disparity_to_depth,
    fisheye_to_linear_calib,
    rectify_stereo_pair,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export depth from stereo using Foundation Stereo"
    )
    parser.add_argument("--vrs", required=True, help="Path to VRS file")
    parser.add_argument(
        "--mps", required=True, help="Path to MPS directory (contains slam/ subfolder)"
    )
    parser.add_argument(
        "--stereo_model", required=True, help="Path to Foundation Stereo checkpoint"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--max_frames", type=int, default=0, help="Max frames to process (0 = all)"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every Nth VRS frame (default 1)"
    )
    parser.add_argument(
        "--no_images", action="store_true", help="Skip writing PNG images"
    )
    return parser.parse_args()


def load_foundation_stereo(ckpt_path, valid_iters=32):
    """Load Foundation Stereo model."""
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg.valid_iters = valid_iters

    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval()

    print(f"Foundation Stereo loaded (epoch {ckpt['epoch']}, vit={cfg.vit_size})")
    return model, cfg


def run_foundation_stereo(model, left_rect, right_rect, cfg):
    """Run Foundation Stereo inference on rectified grayscale images.

    Input preprocessing:
    - Grayscale replicated to 3 channels
    - Float32 in range [0, 255] (NOT normalized to [0, 1])
    - BCHW format [1, 3, H, W]
    - Padded to multiple of 32
    """
    left_rgb = np.stack([left_rect] * 3, axis=-1)  # [H, W, 3]
    right_rgb = np.stack([right_rect] * 3, axis=-1)

    left_t = torch.from_numpy(left_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)
    right_t = torch.from_numpy(right_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)

    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_p, right_p = padder.pad(left_t, right_t)

    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            disp = model.forward(left_p, right_p, iters=cfg.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float()).cpu().numpy().squeeze()
    return disp


def build_frame_json(index, T_world_camera, camera_calib, timestamp_ns):
    """Build JSON dict for a single frame.

    SE3 quaternion order is XYZW convention.
    """
    # Extract quaternion and translation from SE3: [w, x, y, z, tx, ty, tz]
    qw, qx, qy, qz, tx, ty, tz = [
        float(x) for x in T_world_camera.to_quat_and_translation()[0]
    ]

    # Camera model parameters [fx, fy, cx, cy]
    params = camera_calib.get_projection_params()

    return {
        "index": index,
        "T_world_camera": {
            "QuaternionXYZW": [qx, qy, qz, qw],
            "Translation": [tx, ty, tz],
        },
        "camera": {
            "ModelName": "Linear:fu,fv,u0,v0",
            "Parameters": [float(p) for p in params],
        },
        "frameTimestampNs": int(timestamp_ns),
    }


def write_to_disk(index, depth_dir, images_dir, rect_image, depth_map):
    """Write image and depth PNGs.

    Image: uint8 grayscale PNG into rectified_images/
    Depth: uint16 PNG in millimeters, clamped to [0, 65535] into depth/
    """
    img_path = os.path.join(images_dir, f"image_{index:08d}.png")
    Image.fromarray(rect_image.astype(np.uint8)).save(img_path)

    depth_mm = (depth_map * 1000).astype(np.int32)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    depth_path = os.path.join(depth_dir, f"depth_{index:08d}.png")
    Image.fromarray(depth_mm).save(depth_path)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, "depth")
    images_dir = os.path.join(args.output_dir, "rectified_images")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Load VRS
    print(f"Loading VRS: {args.vrs}")
    vrs = data_provider.create_vrs_data_provider(args.vrs)
    assert vrs is not None, f"Failed to load VRS file: {args.vrs}"

    left_label = "slam-front-left"
    right_label = "slam-front-right"
    left_stream = vrs.get_stream_id_from_label(left_label)
    right_stream = vrs.get_stream_id_from_label(right_label)
    assert left_stream is not None, "Could not find slam-front-left stream"
    assert right_stream is not None, "Could not find slam-front-right stream"

    num_left = vrs.get_num_data(left_stream)
    num_right = vrs.get_num_data(right_stream)
    num_frames = min(num_left, num_right)
    if num_left != num_right:
        print(
            f"  WARNING: Left/right frame count mismatch ({num_left} vs {num_right}), using {num_frames}"
        )
    print(f"  {num_frames} frames available")

    # Load MPS data via MpsDataPathsProvider
    print(f"Loading MPS data: {args.mps}")
    mps_data_paths = MpsDataPathsProvider(args.mps).get_data_paths()
    mps = MpsDataProvider(mps_data_paths)
    print("  Trajectory and online calibration loaded")

    # Discover camera labels in online calibration
    first_online_calib = mps.get_online_calibration(
        vrs.get_image_data_by_index(left_stream, 0)[1].capture_timestamp_ns,
        TimeQueryOptions.CLOSEST,
    )
    assert first_online_calib is not None, "No online calibration data found"

    # Load Foundation Stereo
    print(f"Loading Foundation Stereo: {args.stereo_model}")
    model, cfg = load_foundation_stereo(args.stereo_model)

    # Process frames
    max_output = args.max_frames if args.max_frames > 0 else num_frames
    print(
        f"\nProcessing up to {max_output} output frames from {num_frames} VRS frames..."
    )

    json_frames = []
    executor = ThreadPoolExecutor(max_workers=2)
    write_futures = []
    output_idx = 0
    loop_start = time.time()

    for i in range(0, num_frames, args.stride):
        if output_idx >= max_output:
            break

        # 1. Load images (right image looked up by left timestamp for robustness)
        left_data, left_record = vrs.get_image_data_by_index(left_stream, i)
        timestamp_ns = left_record.capture_timestamp_ns
        right_data, right_record = vrs.get_image_data_by_time_ns(
            right_stream, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )
        left_image = left_data.to_numpy_array()
        right_image = right_data.to_numpy_array()

        # Stereo pairs should have nearly identical timestamps; allow up to 1ms
        timestamp_diff_ns = abs(
            left_record.capture_timestamp_ns - right_record.capture_timestamp_ns
        )
        if timestamp_diff_ns > 1_000_000:
            print(
                f"\n  WARNING: Timestamp mismatch at index {i}: "
                f"left={left_record.capture_timestamp_ns} right={right_record.capture_timestamp_ns} "
                f"(diff={timestamp_diff_ns}ns), skipping"
            )
            continue

        # 2. Get pose from trajectory (SE3 interpolation)
        pose = mps.get_interpolated_closed_loop_pose(timestamp_ns)
        if pose is None:
            continue
        T_world_device = pose.transform_world_device

        # 3. Get online calibration (closest timestamp)
        online_calib = mps.get_online_calibration(
            timestamp_ns, TimeQueryOptions.CLOSEST
        )
        if online_calib is None:
            continue
        left_calib = online_calib.get_camera_calib(left_label)
        right_calib = online_calib.get_camera_calib(right_label)
        if left_calib is None or right_calib is None:
            continue

        # 4. Compute T_lr from online extrinsics
        # get_transform_device_camera returns T_device_camera
        # We need T_cam_device = T_device_camera.inverse()
        t0 = time.time()
        T_leftCam_device = left_calib.get_transform_device_camera().inverse()
        T_rightCam_device = right_calib.get_transform_device_camera().inverse()
        T_leftCam_rightCam = T_leftCam_device @ T_rightCam_device.inverse()

        # 5. Get image dimensions
        img_h, img_w = left_image.shape[:2]

        # 6. Create shared rectified pinhole camera from left camera only.
        # Both images must use the same intrinsics for correct stereo rectification.
        shared_linear = fisheye_to_linear_calib(
            left_calib,
            focal_scale=1.25,
            output_width=img_w,
            output_height=img_h,
            use_original_pp=True,
        )

        # 7. Compute rectification rotations
        R_left_rect, R_right_rect = create_scanline_rectified_cameras(
            T_leftCam_device, T_rightCam_device
        )

        # 8. Rectify
        left_rect, right_rect = rectify_stereo_pair(
            left_image,
            right_image,
            left_calib,
            right_calib,
            shared_linear,
            shared_linear,
            R_left_rect,
            R_right_rect,
            interpolation=InterpolationMethod.BILINEAR,
        )

        # 9. Foundation Stereo inference
        disparity_map = run_foundation_stereo(model, left_rect, right_rect, cfg)

        # 10. Disparity → Depth
        baseline = float(np.linalg.norm(T_leftCam_rightCam.translation()))
        focal_length = float(shared_linear.get_projection_params()[0])  # fx
        depth_map = disparity_to_depth(disparity_map, baseline, focal_length)

        # 11. Compute world pose of rectified camera
        T_world_rectCam = compute_T_world_rectCam(
            T_world_device, T_leftCam_device, R_left_rect
        )

        # 12. Build JSON frame entry
        json_frames.append(
            build_frame_json(output_idx, T_world_rectCam, shared_linear, timestamp_ns)
        )

        # 13. Write images (async via thread pool)
        if not args.no_images:
            future = executor.submit(
                write_to_disk, output_idx, depth_dir, images_dir, left_rect, depth_map
            )
            write_futures.append(future)

        elapsed = time.time() - t0
        wall = time.time() - loop_start
        done = output_idx + 1
        fps = done / wall
        eta = (max_output - done) / fps if fps > 0 else 0
        pct = done / max_output * 100
        print(
            f"\r  [{done}/{max_output}] {pct:5.1f}%  "
            f"frame {elapsed:.2f}s  {fps:.1f} fps  "
            f"ETA {int(eta // 60)}m{int(eta % 60):02d}s",
            end="",
            flush=True,
        )
        output_idx += 1

    print()  # newline after progress

    # Wait for all writes to complete
    for future in write_futures:
        future.result()
    executor.shutdown(wait=True)

    # Save frames.json
    json_path = os.path.join(args.output_dir, "pinhole_camera_parameters.json")
    with open(json_path, "w") as f:
        json.dump(json_frames, f, indent=2)
    print(f"\nWrote {len(json_frames)} frames to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
