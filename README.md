


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



# Mamba-Based Progressive-Recovery Framework for Multimodal Low Light Image Enhancement,



## 1. The Code


## Structure

- `mamba.py`: Core Mamba module.
- `modules.py`: Components for illumination corruption restoration (IAI, DSFS, IISSM, MambaUNet).
- `fusion.py`: Components for multimodal fusion enhancement (MMB, MultimodalFusion, etc.).
- `losses.py`: Loss functions (LossA, LossB) and helpers.
- `utils2.py`: Utility functions like bright_channel.
- `model.py`: Main model classes (IlluminationEstimator, ProposedMethod).
- `train.py`: Example training script.
  
## Usage
Install dependencies (PyTorch, etc.), then run `train.py` with your dataset.

There are more .py files for your help. For instance:
`utilities.py`: This file provides utility functions for quality metrics (PSNR, SSIM), feature visualization, and model summary (FLOPs and parameters) in low-light enhancement.
**Key Functions**
  Quality Metrics

calculate_psnr(img1, img2, border=0): PSNR for 8-bit images.
PSNR(img1, img2): Simplified PSNR for [0,1] range images.
calculate_ssim(img1, img2, border=0): SSIM for grayscale/color images.
ssim(img1, img2): Core SSIM with Gaussian filtering.


**Visualization**

visualization(feature, save_path, type='max', colormap=cv2.COLORMAP_JET): Create heatmap from PyTorch feature map.

**Model Analysis
**
my_summary(test_model, H=256, W=256, C=3, N=1): Print model summary, FLOPs (GMac), and parameters (requires fvcore).

**Dependencies**

NumPy, OpenCV, Math, PDB.
fvcore (for FlopCountAnalysis): pip install fvcore.




## 2.  Support
For support, email 20110720115@fudan.edu.cn


## 3. Citation


If you use this code, please cite the original paper:

```bibtex
@article{mahdizadeh2025mamba,
  author={M. Mahdizadeh and J. Cao and P. Ye and T. Chen},
  journal={IEEE Transactions on Multimedia},
  title={Mamba-Based Progressive-Recovery Framework for Multimodal Low Light Image Enhancement},
  year={2025},
  doi={10.1109/TMM.2025.3623502},
}



```




