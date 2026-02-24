# Aria Gen2 Stereo Depth Estimation Tutorial

Complete tutorial for computing metric depth maps from Aria Gen2 stereo cameras using stereo rectification and Foundation Stereo neural network.

## Overview

This tutorial demonstrates the full pipeline:
1. Load stereo camera data from Aria Gen2 VRS files
2. Perform stereo rectification on fisheye images
3. Use Foundation Stereo for zero-shot disparity estimation
4. Convert disparity to metric depth
5. Visualize depth as 3D point clouds with Rerun

## Prerequisites

- **CUDA-capable GPU** with 2-4GB VRAM
- **Aria Gen2 VRS file** such as from the Project Aria Gen2 Pilot Dataset [TODO: link]

## Quick Start

### 1. Clone and Set Up

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo.git
cd depth_from_stereo

# Or if already cloned without submodules:
git submodule update --init

# Create conda environment
conda env create -f environment.yml
conda activate depth_from_stereo
```

This installs:
- Python 3.11
- PyTorch 2.10.0 with CUDA 12.8 support
- Project Aria Tools 2.1.0 with all extras
- Foundation Stereo dependencies (timm, einops, xformers, flash-attn, etc.)
- Rerun SDK for 3D visualization

### 2. Download Foundation Stereo Checkpoint

```bash
# Download checkpoint (~3.2 GB) into the submodule
# See: https://github.com/NVlabs/FoundationStereo for download instructions
# Place model_best_bp2-001.pth and cfg.yaml in FoundationStereo/ckpts/
```

### 3. Run the Export Script

To process an entire recording and export rectified images, depth maps, and camera metadata:

```bash
python export_depth_from_stereo.py \
  --vrs ~/datasets/projectaria_gen2_pilot_dataset/walk_0/video.vrs \
  --mps ~/datasets/projectaria_gen2_pilot_dataset/walk_0/mps \
  --stereo_model ./FoundationStereo/ckpts/model_best_bp2-001.pth \
  --output_dir ./output/walk_0
```

Optional flags:
- `--max_frames N` — Limit to N output frames (0 = all)
- `--stride N` — Process every Nth VRS frame (default 1)
- `--no_images` — Skip writing PNG images, only produce `pinhole_camera_parameters.json`

The output directory will contain:
- `rectified_images/image_XXXXXXXX.png` — Rectified left camera images (uint8 grayscale)
- `depth/depth_XXXXXXXX.png` — Depth maps as uint16 PNGs in millimeters
- `pinhole_camera_parameters.json` — Per-frame camera intrinsics and world poses

### 4. Run the Tutorial Notebook

Open the Jupyter notebook using your preferred notebook viewer.

## Tutorial Contents

The tutorial covers:

1. **Environment Setup** - Import libraries and verify GPU
2. **VRS Data Loading** - Load stereo cameras and calibration
3. **Stereo Rectification** - Transform fisheye to pinhole with horizontal epipolar lines
4. **Foundation Stereo Inference** - Compute disparity map
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
- Project Aria Tools: See [Project Aria Tools LICENSE](https://github.com/facebookresearch/projectaria_tools/blob/main/LICENSE)

## Support

For issues related to:
- **Tutorial**: Open an issue in this repository
- **Foundation Stereo**: See [Foundation Stereo Issues](https://github.com/NVlabs/FoundationStereo/issues)
- **Project Aria Tools**: See [Project Aria Tools Issues](https://github.com/facebookresearch/projectaria_tools/issues)


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
projectaria_gen2_depth_from_stereo is Apache 2.0 licensed, as found in the LICENSE file.
