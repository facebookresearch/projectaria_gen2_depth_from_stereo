# Aria Gen2 Stereo Depth Estimation Tutorial

Complete tutorial for computing metric depth maps from Aria Gen2 stereo cameras using stereo rectification and a selectable stereo backend.

## Overview

This tutorial demonstrates the full pipeline:
1. Load stereo camera data from Aria Gen2 VRS files
2. Perform stereo rectification on fisheye images
3. Use Foundation Stereo or WAFT-Stereo for zero-shot disparity estimation
4. Convert disparity to metric depth
5. Visualize depth as 3D point clouds with Rerun

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

### 2b. Optional: Set Up WAFT-Stereo

WAFT-Stereo is supported as an optional backend and is included as a git submodule:

```bash
git submodule update --init WAFT-Stereo
```

Install the additional WAFT dependencies:

```bash
pip install peft yacs termcolor accelerate
```

Download the required checkpoints:

- WAFT checkpoint:
  - `WAFT-Stereo/ckpts/SynLarge/DAv2L-5.pth`
- Depth-Anything-V2 checkpoint:
  - `WAFT-Stereo/depth-anything-ckpts/depth_anything_v2_vitl.pth`

See:
- https://github.com/princeton-vl/WAFT-Stereo
- https://huggingface.co/MemorySlices/WAFT-Stereo
- https://huggingface.co/depth-anything/Depth-Anything-V2-Large

### 3. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate depth_from_stereo
```

This installs:
- Python 3.11
- PyTorch 2.10.0 with CUDA 12.8 support
- Project Aria Tools 2.1.0 with all extras
- Foundation Stereo dependencies (timm, einops, xformers, flash-attn, etc.)
- Rerun SDK for 3D visualization

### 4. Run the Export Script

**Note:** Make sure the `depth_from_stereo` conda environment is activated before running.

To process an entire recording and export rectified images, depth maps, and camera metadata with the default Foundation Stereo backend:

```bash
python export_depth_from_stereo.py \
  --vrs ~/datasets/projectaria_gen2_pilot_dataset/walk_0/video.vrs \
  --mps ~/datasets/projectaria_gen2_pilot_dataset/walk_0/mps \
  --stereo_model ./FoundationStereo/ckpts/model_best_bp2-001.pth \
  --stereo_backend foundation \
  --output_dir ./output/walk_0
```

To use WAFT-Stereo instead:

```bash
python export_depth_from_stereo.py \
  --vrs ~/datasets/projectaria_gen2_pilot_dataset/walk_0/video.vrs \
  --mps ~/datasets/projectaria_gen2_pilot_dataset/walk_0/mps \
  --stereo_model ./WAFT-Stereo/ckpts/SynLarge/DAv2L-5.pth \
  --stereo_backend waft \
  --waft_config ./WAFT-Stereo/configs/SynLarge/DAv2L-5.yaml \
  --waft_tile_height 544 \
  --waft_tile_width 960 \
  --waft_factor_list 1.0 \
  --output_dir ./output/walk_0_waft
```

Optional flags:
- `--max_frames N` — Limit to N output frames (0 = all)
- `--stride N` — Process every Nth VRS frame (default 1)
- `--no_images` — Skip writing PNG images, only produce `pinhole_camera_parameters.json`
- `--stereo_backend {foundation,waft}` — Choose which stereo model family to use
- `--backend {torch,tensorrt}` — Foundation Stereo runtime backend only
- `--lr_check` — Run Foundation Stereo left-right consistency checking and write `masks/mask_XXXXXXXX.png`
- `--zero_inconsistent_depth` — With `--lr_check`, zero inconsistent depth pixels before writing `depth/*.png`
- `--waft_tile_height/--waft_tile_width` — WAFT tiled inference crop size
- `--waft_factor_list` — WAFT inference scale factors, e.g. `0.5,1.0`

Notes:
- `--lr_check` is currently supported only for `--stereo_backend foundation`
- `--zero_inconsistent_depth` requires `--lr_check`
- `--backend tensorrt` applies only to Foundation Stereo

The output directory will contain:
- `rectified_images/image_XXXXXXXX.png` — Rectified left camera images (uint8 grayscale)
- `depth/depth_XXXXXXXX.png` — Depth maps as uint16 PNGs in millimeters
- `masks/mask_XXXXXXXX.png` — Optional LR-consistency masks (255 = consistent, 0 = inconsistent)
- `pinhole_camera_parameters.json` — Per-frame camera intrinsics, `T_world_camera`, and `T_device_rectCam`

### 5. Run the Tutorial Notebook

Open the Jupyter notebook using your preferred notebook viewer.

## Tutorial Contents

The tutorial covers:

1. **Environment Setup** - Import libraries and verify GPU
2. **VRS Data Loading** - Load stereo cameras and calibration
3. **Stereo Rectification** - Transform fisheye to pinhole with horizontal epipolar lines
4. **Stereo Inference** - Compute disparity map with Foundation Stereo or WAFT-Stereo
5. **Depth Conversion** - Convert disparity to metric depth
6. **3D Visualization** - Interactive point cloud with Rerun

## Configuration

Update these paths in the notebook/script:

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

- `depth_from_stereo.ipynb` - Main tutorial notebook
- `stereo_utils.py` - Helper functions for rectification and depth conversion

## Performance

For faster inference:
- Use TensorRT for 3-6x speedup
- Reduce image resolution
- Reduce refinement iterations (quality tradeoff)

NOTE: The results of this tutorial are not guaranteed to exactly match depth maps from other pipelines such as the Gen 2 Pilot Dataset.

## Resources

- [Project Aria Tools Documentation](https://facebookresearch.github.io/projectaria_tools/) - Aria API reference
- [Foundation Stereo GitHub](https://github.com/NVlabs/FoundationStereo) - Model repository
- [WAFT-Stereo GitHub](https://github.com/princeton-vl/WAFT-Stereo) - Optional stereo backend
- [Rerun Documentation](https://www.rerun.io/docs) - 3D visualization guide

## Citation

If you use this tutorial in your research, please cite:

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


## License

This tutorial follows the licensing of the underlying tools:
- Foundation Stereo: See [Foundation Stereo LICENSE](https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE)
- WAFT-Stereo: See [WAFT-Stereo LICENSE](https://github.com/princeton-vl/WAFT-Stereo/blob/main/LICENSE)
- Project Aria Tools: See [Project Aria Tools LICENSE](https://github.com/facebookresearch/projectaria_tools/blob/main/LICENSE)

## Support

For issues related to:
- **Tutorial**: Open an issue in this repository
- **Foundation Stereo**: See [Foundation Stereo Issues](https://github.com/NVlabs/FoundationStereo/issues)
- **WAFT-Stereo**: See [WAFT-Stereo Issues](https://github.com/princeton-vl/WAFT-Stereo/issues)
- **Project Aria Tools**: See [Project Aria Tools Issues](https://github.com/facebookresearch/projectaria_tools/issues)


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
projectaria_gen2_depth_from_stereo is Apache 2.0 licensed, as found in the LICENSE file.
