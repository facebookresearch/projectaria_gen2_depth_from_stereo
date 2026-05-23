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

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

# Foundation Stereo (git submodule at ./FoundationStereo). Imports stay lazy so
# --no_images can regenerate only metadata without requiring torch/checkpoints.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOUNDATION_STEREO_PATH = os.path.join(SCRIPT_DIR, "FoundationStereo")


class SingleEngineTrtRunner:
    """TRT runner wrapping a single engine, matching TrtRunner.forward() API."""

    def __init__(self, model_path: str):
        if os.path.isdir(model_path):
            engine_path = os.path.join(model_path, "tensorrt.engine")
        else:
            engine_path = model_path
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"No TensorRT engine found at {engine_path}. "
                "Pass either a .engine file or a directory containing tensorrt.engine."
            )

        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                "TensorRT backend requires NVIDIA TensorRT Python bindings. "
                "Install TensorRT separately for your CUDA/driver/runtime before "
                "using --backend tensorrt."
            ) from e

        self.engine_path = engine_path
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(self._trt_logger).deserialize_cuda_engine(
                f.read()
            )
        self._context = self._engine.create_execution_context()

    def _trt_dtype_to_torch(self, dt):
        import tensorrt as trt
        import torch

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f"Unsupported TRT dtype: {dt}")
        return mapping[dt]

    def _get_io_names(self, mode):
        import tensorrt as trt  # noqa: F811

        return [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == mode
        ]

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        import tensorrt as trt
        import torch

        inputs = {"left": left, "right": right}

        for name, tensor in inputs.items():
            expected = self._trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                inputs[name] = tensor.to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            self._context.set_input_shape(name, tuple(inputs[name].shape))

        outputs = {}
        for name in self._get_io_names(trt.TensorIOMode.OUTPUT):
            shape = tuple(self._context.get_tensor_shape(name))
            dtype = self._trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in inputs.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        ok = self._context.execute_async_v3(stream)
        assert ok, "TRT engine execution failed"

        return outputs["disp"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export depth from stereo using Foundation Stereo"
    )
    parser.add_argument("--vrs", required=True, help="Path to VRS file")
    parser.add_argument(
        "--mps", required=True, help="Path to MPS directory (contains slam/ subfolder)"
    )
    parser.add_argument(
        "--stereo_model",
        default="",
        help="Path to Foundation Stereo checkpoint (.pth), TensorRT .engine file, or directory containing tensorrt.engine",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "tensorrt"],
        help="Inference runtime backend for Foundation Stereo only: 'torch' (default) or 'tensorrt'",
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
    parser.add_argument(
        "--lr_check",
        action="store_true",
        help="Enable left-right disparity consistency check",
    )
    parser.add_argument(
        "--lr_threshold",
        type=float,
        default=1.0,
        help="LR consistency threshold in pixels (default 1.0)",
    )
    parser.add_argument(
        "--zero_inconsistent_depth",
        action="store_true",
        help="When used with --lr_check, zero depth at LR-inconsistent pixels before writing",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.no_images and not args.stereo_model:
        raise ValueError("--stereo_model is required unless --no_images is set")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be >= 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.lr_threshold <= 0:
        raise ValueError("--lr_threshold must be > 0")
    if args.zero_inconsistent_depth and not args.lr_check:
        raise ValueError("--zero_inconsistent_depth requires --lr_check")


def load_runtime_dependencies():
    """Import Project Aria and local stereo helpers after CLI parsing."""
    global data_provider
    global InterpolationMethod
    global MpsDataPathsProvider
    global MpsDataProvider
    global TimeDomain
    global TimeQueryOptions
    global compute_T_device_rectCam
    global compute_T_world_rectCam
    global create_scanline_rectified_cameras
    global disparity_to_depth
    global fisheye_to_linear_calib
    global rectify_stereo_pair

    from projectaria_tools.core import data_provider as _data_provider
    from projectaria_tools.core.image import InterpolationMethod as _InterpolationMethod
    from projectaria_tools.core.mps import (
        MpsDataPathsProvider as _MpsDataPathsProvider,
        MpsDataProvider as _MpsDataProvider,
    )
    from projectaria_tools.core.sensor_data import (
        TimeDomain as _TimeDomain,
        TimeQueryOptions as _TimeQueryOptions,
    )
    from stereo_utils import (
        compute_T_device_rectCam as _compute_T_device_rectCam,
        compute_T_world_rectCam as _compute_T_world_rectCam,
        create_scanline_rectified_cameras as _create_scanline_rectified_cameras,
        disparity_to_depth as _disparity_to_depth,
        fisheye_to_linear_calib as _fisheye_to_linear_calib,
        rectify_stereo_pair as _rectify_stereo_pair,
    )

    data_provider = _data_provider
    InterpolationMethod = _InterpolationMethod
    MpsDataPathsProvider = _MpsDataPathsProvider
    MpsDataProvider = _MpsDataProvider
    TimeDomain = _TimeDomain
    TimeQueryOptions = _TimeQueryOptions
    compute_T_device_rectCam = _compute_T_device_rectCam
    compute_T_world_rectCam = _compute_T_world_rectCam
    create_scanline_rectified_cameras = _create_scanline_rectified_cameras
    disparity_to_depth = _disparity_to_depth
    fisheye_to_linear_calib = _fisheye_to_linear_calib
    rectify_stereo_pair = _rectify_stereo_pair


def load_foundation_stereo(model_path, backend="torch", valid_iters=32):
    """Load Foundation Stereo model.

    Returns (model_or_runner, cfg_or_None).
    """
    if backend == "tensorrt":
        runner = SingleEngineTrtRunner(model_path)
        print(f"Foundation Stereo TensorRT engine loaded from {runner.engine_path}")
        return runner, None

    import torch

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
    if FOUNDATION_STEREO_PATH not in sys.path:
        sys.path.insert(0, FOUNDATION_STEREO_PATH)

    from core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf

    cfg_path = os.path.join(os.path.dirname(model_path), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg.valid_iters = valid_iters

    model = FoundationStereo(cfg)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
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
    - Padded to multiple of 32 (torch backend only)
    """
    import torch

    left_rgb = np.stack([left_rect] * 3, axis=-1)  # [H, W, 3]
    right_rgb = np.stack([right_rect] * 3, axis=-1)

    left_t = torch.from_numpy(left_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)
    right_t = torch.from_numpy(right_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)

    if cfg is None:
        # TensorRT backend
        with torch.no_grad():
            disp = model.forward(left_t, right_t)
        return disp.float().cpu().numpy().squeeze()

    # PyTorch backend
    from core.utils.utils import InputPadder

    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_p, right_p = padder.pad(left_t, right_t)

    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            disp = model.forward(left_p, right_p, iters=cfg.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float()).cpu().numpy().squeeze()
    return disp


def run_lr_consistency(model, left_rect, right_rect, cfg, threshold=1.0):
    """Run LR consistency check: two passes of Foundation Stereo.

    Returns (disparity_map, consistent_mask) where consistent_mask is bool [H, W].
    """
    disp_lr = run_foundation_stereo(model, left_rect, right_rect, cfg)

    right_flipped = np.ascontiguousarray(right_rect[:, ::-1])
    left_flipped = np.ascontiguousarray(left_rect[:, ::-1])
    disp_rl = run_foundation_stereo(model, right_flipped, left_flipped, cfg)
    disp_rl = np.ascontiguousarray(disp_rl[:, ::-1])

    h, w = disp_lr.shape
    x_coords = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    x_in_right_f = x_coords - disp_lr
    out_of_bounds = (x_in_right_f < 0) | (x_in_right_f > w - 1)
    x_in_right = np.clip(x_in_right_f, 0, w - 1).astype(np.int32)
    rows = np.arange(h)[:, None].repeat(w, axis=1)
    disp_rl_at_match = disp_rl[rows, x_in_right]
    consistent = (np.abs(disp_lr - disp_rl_at_match) < threshold) & ~out_of_bounds

    return disp_lr, consistent


def transform_to_json(transform):
    """Convert an SE3 transform to the JSON pose representation."""
    # SE3 returns [w, x, y, z, tx, ty, tz]; JSON uses XYZW quaternion order.
    qw, qx, qy, qz, tx, ty, tz = [
        float(x) for x in transform.to_quat_and_translation()[0]
    ]
    return {
        "QuaternionXYZW": [qx, qy, qz, qw],
        "Translation": [tx, ty, tz],
    }


def build_frame_json(
    index, T_world_camera, T_device_rectCam, camera_calib, timestamp_ns
):
    """Build JSON dict for a single frame.

    JSON quaternion order is XYZW convention.
    """
    # Camera model parameters [fx, fy, cx, cy]
    params = camera_calib.get_projection_params()

    return {
        "index": index,
        "T_world_camera": transform_to_json(T_world_camera),
        "T_device_rectCam": transform_to_json(T_device_rectCam),
        "camera": {
            "ModelName": "Linear:fu,fv,u0,v0",
            "Parameters": [float(p) for p in params],
        },
        "frameTimestampNs": int(timestamp_ns),
    }


def write_to_disk(
    index, depth_dir, images_dir, rect_image, depth_map, masks_dir=None, mask=None
):
    """Write image and depth PNGs, and optionally a consistency mask.

    Image: uint8 grayscale PNG into rectified_images/
    Depth: uint16 PNG in millimeters, clamped to [0, 65535] into depth/
    Mask: uint8 PNG (255=consistent, 0=inconsistent) into masks/
    """
    img_path = os.path.join(images_dir, f"image_{index:08d}.png")
    Image.fromarray(rect_image.astype(np.uint8)).save(img_path)

    depth_mm = np.nan_to_num(depth_map * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    depth_path = os.path.join(depth_dir, f"depth_{index:08d}.png")
    Image.fromarray(depth_mm).save(depth_path)

    if masks_dir is not None and mask is not None:
        mask_path = os.path.join(masks_dir, f"mask_{index:08d}.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)


def main():
    args = parse_args()
    validate_args(args)
    load_runtime_dependencies()

    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, "depth")
    images_dir = os.path.join(args.output_dir, "rectified_images")
    if not args.no_images:
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

    masks_dir = None
    if args.lr_check and not args.no_images:
        masks_dir = os.path.join(args.output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

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

    # Load stereo model only when depth/images are being regenerated.
    if args.no_images:
        print("Metadata-only mode (--no_images): skipping stereo model/inference")
        model = None
        cfg = None
    else:
        print(f"Loading Foundation Stereo: {args.stereo_model}")
        model, cfg = load_foundation_stereo(args.stereo_model, backend=args.backend)

    # Process frames
    available_outputs = (num_frames + args.stride - 1) // args.stride
    max_output = (
        min(args.max_frames, available_outputs)
        if args.max_frames > 0
        else available_outputs
    )
    print(
        f"\nProcessing up to {max_output} output frames from {num_frames} "
        f"VRS frames (stride {args.stride})..."
    )

    json_frames = []
    executor = ThreadPoolExecutor(max_workers=2)
    write_futures = []
    output_idx = 0
    loop_start = time.time()

    for i in range(0, num_frames, args.stride):
        if output_idx >= max_output:
            break

        # 1. Load timestamps/images (right frame looked up by left timestamp for
        # robustness). In metadata-only mode we avoid converting image payloads
        # to numpy and skip all stereo inference.
        left_data, left_record = vrs.get_image_data_by_index(left_stream, i)
        timestamp_ns = left_record.capture_timestamp_ns
        right_data, right_record = vrs.get_image_data_by_time_ns(
            right_stream, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )
        if args.no_images:
            left_image = None
            right_image = None
        else:
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
        if args.no_images:
            img_w, img_h = [int(v) for v in left_calib.get_image_size()]
        else:
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

        consistency_mask = None
        if not args.no_images:
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

            # 9. Stereo inference
            if args.lr_check:
                disparity_map, consistency_mask = run_lr_consistency(
                    model, left_rect, right_rect, cfg, threshold=args.lr_threshold
                )
            else:
                disparity_map = run_foundation_stereo(model, left_rect, right_rect, cfg)

            # 10. Disparity → Depth
            baseline = float(np.linalg.norm(T_leftCam_rightCam.translation()))
            focal_length = float(shared_linear.get_projection_params()[0])  # fx
            depth_map = disparity_to_depth(disparity_map, baseline, focal_length)

            # Optionally zero out inconsistent depth pixels before writing.
            if consistency_mask is not None and args.zero_inconsistent_depth:
                depth_map[~consistency_mask] = 0.0

        # 11. Compute poses of the rectified camera
        T_world_rectCam = compute_T_world_rectCam(
            T_world_device, T_leftCam_device, R_left_rect
        )
        T_device_rectCam = compute_T_device_rectCam(T_leftCam_device, R_left_rect)

        # 12. Build JSON frame entry
        json_frames.append(
            build_frame_json(
                output_idx,
                T_world_rectCam,
                T_device_rectCam,
                shared_linear,
                timestamp_ns,
            )
        )

        # 13. Write images (async via thread pool)
        if not args.no_images:
            future = executor.submit(
                write_to_disk,
                output_idx,
                depth_dir,
                images_dir,
                left_rect,
                depth_map,
                masks_dir,
                consistency_mask,
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
