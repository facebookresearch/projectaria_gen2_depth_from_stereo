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

### 1. Create Conda Environment

```bash
# Create environment with all dependencies
conda env create -f environment.yml

# Activate the environment
conda activate foundation_stereo
```

This single command installs:
- Python 3.11
- PyTorch 2.4.1 with CUDA 12.8 support
- Project Aria Tools 2.1.0 with all extras
- Foundation Stereo dependencies (timm, einops, xformers, flash-attn, etc.)
- Rerun SDK for 3D visualization

**Note:** Installation may take 10-20 minutes depending on your internet connection.

### 2. Download Foundation Stereo Model

Download the pretrained model weights:

```bash
# Create directory
mkdir -p FoundationStereo/pretrained_models/23-51-11

# Download model (you'll need to get this from Foundation Stereo repository)
# See: https://github.com/NVlabs/FoundationStereo
```

### 3. Run the Tutorial

Open the Jupyter notebook using your preferred notebook viewer.

## Tutorial Contents

The tutorial covers:

1. **Environment Setup** - Import libraries and verify GPU
2. **VRS Data Loading** - Load stereo cameras and calibration
3. **Stereo Rectification** - Transform fisheye to pinhole with horizontal epipolar lines
4. **Foundation Stereo Inference** - Compute disparity map
5. **Depth Conversion** - Convert disparity to metric depth
6. **3D Visualization** - Interactive point cloud with Rerun
7. **Summary** - Performance metrics and next steps

## Configuration

Update these paths in the notebook/script:

```python
# Path to your Aria Gen2 VRS file
VRS_FILE_PATH = "path/to/your/aria_recording.vrs"

# Path to Foundation Stereo checkpoint (if different)
FOUNDATION_STEREO_CKPT = "./FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"

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
