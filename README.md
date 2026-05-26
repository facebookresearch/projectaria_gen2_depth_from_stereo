# Aria Gen2 Depth From Stereo

Utilities for computing metric depth maps from Aria Gen2 front-facing stereo cameras using stereo rectification and Foundation Stereo.

## Overview

This repository provides:
- `export_depth_from_stereo.py` - A command-line utility for exporting rectified stereo images, metric depth maps, optional consistency masks, and pinhole camera metadata from Aria Gen2 VRS + MPS data.
- `depth_from_stereo.ipynb` - A notebook tutorial that walks through the same pipeline step by step.

The depth export pipeline:
1. Load stereo camera data from Aria Gen2 VRS files
2. Perform stereo rectification on fisheye images
3. Use Foundation Stereo for zero-shot disparity estimation
4. Convert disparity to metric depth
5. Write depth maps and camera metadata for downstream use

## Prerequisites

- **CUDA-capable GPU** with 2-4GB VRAM
- **Aria Gen2 VRS file** such as from the Project Aria Gen2 Pilot Dataset:

VRS: https://www.projectaria.com/async/sample/download/?bucket=core&filename=aria_gen2_sample_data_1.vrs
MPS: https://www.projectaria.com/async/sample/download/?bucket=core&filename=aria_gen2_sample_data_1_mps_output_dec_2025.zip

## Quick Start

### 1. Clone Repository

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo.git
cd projectaria_gen2_depth_from_stereo

# Or if already cloned without submodules:
git submodule update --init
```

### 2. Download Foundation Stereo Checkpoint

Start this download first — the checkpoint is ~3.2 GB and can download while conda installs in the next step.

```bash
# Download checkpoint (~3.2 GB) into the submodule
# See: https://github.com/NVlabs/FoundationStereo for download instructions
# Place model_best_bp2-001.pth and cfg.yaml in FoundationStereo/ckpts/
```

### 3. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate depth_from_stereo
```

This installs:
- Python 3.11
- PyTorch 2.10.0 with CUDA 12.8 support
- Project Aria Tools 2.1.0 with all extras
- Foundation Stereo dependencies (timm, einops, xformers, etc.)
- Rerun SDK for 3D visualization

### 4. Run the Export Utility

**Note:** Make sure the `depth_from_stereo` conda environment is activated before running.

To process a recording with the default PyTorch backend:

```bash
python export_depth_from_stereo.py \
  --vrs ~/datasets/projectaria_gen2_pilot_dataset/walk_0/video.vrs \
  --mps ~/datasets/projectaria_gen2_pilot_dataset/walk_0/mps \
  --stereo_model ./FoundationStereo/ckpts/model_best_bp2-001.pth \
  --output_dir ./output/walk_0
```

To process a recording with a TensorRT engine:

```bash
python export_depth_from_stereo.py \
  --vrs ~/datasets/projectaria_gen2_pilot_dataset/walk_0/video.vrs \
  --mps ~/datasets/projectaria_gen2_pilot_dataset/walk_0/mps \
  --backend tensorrt \
  --stereo_model ./FoundationStereo/ckpts/tensorrt.engine \
  --output_dir ./output/walk_0
```

Optional flags:
- `--max_frames N` — Limit to N output frames (0 = all)
- `--stride N` — Process every Nth VRS frame (default 1)
- `--no_images` — Skip writing PNG images, only produce `pinhole_camera_parameters.json`
- `--backend {torch,tensorrt}` — Foundation Stereo runtime backend only. With `tensorrt`, pass either `--stereo_model /path/to/tensorrt.engine` or a directory containing `tensorrt.engine`.
- `--lr_check` — Run Foundation Stereo left-right consistency checking and write `masks/mask_XXXXXXXX.png`
- `--zero_inconsistent_depth` — With `--lr_check`, zero inconsistent depth pixels before writing `depth/*.png`

Notes:
- `--zero_inconsistent_depth` requires `--lr_check`
- `--backend tensorrt` applies only to Foundation Stereo and requires NVIDIA TensorRT Python bindings installed separately from the base environment
- The default PyTorch backend may use or download DINOv2 weights through torch hub on first run

The output directory will contain:
- `rectified_images/image_XXXXXXXX.png` — Rectified left camera images (uint8 grayscale)
- `depth/depth_XXXXXXXX.png` — Depth maps as uint16 PNGs in millimeters
- `masks/mask_XXXXXXXX.png` — Optional LR-consistency masks (255 = consistent, 0 = inconsistent)
- `pinhole_camera_parameters.json` — Per-frame camera intrinsics, `T_world_camera`, and `T_device_rectCam`

### 5. Explore the Notebook

Open the Jupyter notebook using your preferred notebook viewer.

## Notebook Tutorial

The notebook covers:

1. **Environment Setup** - Import libraries and verify GPU
2. **VRS Data Loading** - Load stereo cameras and calibration
3. **Stereo Rectification** - Transform fisheye to pinhole with horizontal epipolar lines
4. **Stereo Inference** - Compute disparity map with Foundation Stereo
5. **Depth Conversion** - Convert disparity to metric depth
6. **3D Visualization** - Interactive point cloud with Rerun

## Notebook Configuration

Update these paths in the notebook:

```python
# Path to your Aria Gen2 VRS file
VRS_FILE_PATH = "path/to/your/aria_recording.vrs"

# Optional MPS data directory (set to None to use factory calibration only)
MPS_DIR = "/path/to/sequence/mps/"

# Path to Foundation Stereo checkpoint (if different)
FOUNDATION_STEREO_CKPT = "./FoundationStereo/ckpts/model_best_bp2-001.pth"

# Frame index to process
FRAME_INDEX = 100
```

## Key Files

- `export_depth_from_stereo.py` - Command-line depth export utility
- `depth_from_stereo.ipynb` - Step-by-step notebook tutorial
- `stereo_utils.py` - Helper functions for rectification and depth conversion

## Performance

For faster inference:
- Use TensorRT for 3-6x speedup. TensorRT is optional and host-specific, so it is not installed by `environment.yml`; install NVIDIA TensorRT Python bindings that match your CUDA/driver/runtime before using `--backend tensorrt`.
- Reduce image resolution
- Reduce refinement iterations (quality tradeoff)

NOTE: The exported depth maps are not guaranteed to exactly match depth maps from other pipelines such as the Gen 2 Pilot Dataset.

## Resources

- [Project Aria Tools Documentation](https://facebookresearch.github.io/projectaria_tools/) - Aria API reference
- [Foundation Stereo GitHub](https://github.com/NVlabs/FoundationStereo) - Model repository
- [Rerun Documentation](https://www.rerun.io/docs) - 3D visualization guide

## Citation

If you use this project in your research, please cite:

```bibtex
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```
as well as the Project Aria Gen2 paper:
```bibtex
@article{aria_gen2_egocentric_ai_2025,
  title     = {Aria Gen 2: An Advanced Research Device for Egocentric AI Research},
  author    = {{Project Aria Team at Meta}},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {Meta Reality Labs Research},
}
```

## Support

For issues related to:
- **This repository**: Open an issue in this repository
- **Foundation Stereo**: See [Foundation Stereo Issues](https://github.com/NVlabs/FoundationStereo/issues)
- **Project Aria Tools**: See [Project Aria Tools Issues](https://github.com/facebookresearch/projectaria_tools/issues)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
projectaria_gen2_depth_from_stereo is Apache 2.0 licensed, as found in the LICENSE file.

This project follows the licensing of the underlying tools:
- Foundation Stereo: See [Foundation Stereo LICENSE](https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE)
- Project Aria Tools: See [Project Aria Tools LICENSE](https://github.com/facebookresearch/projectaria_tools/blob/main/LICENSE)
