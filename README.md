


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

- There are more .py files for your help. For instance:
- `utilities.py`: This file provides utility functions for quality metrics (PSNR, SSIM), feature visualization, and model summary (FLOPs and parameters) in low-light enhancement.
**Key Functions**
  Quality Metrics

calculate_psnr(img1, img2, border=0): PSNR for 8-bit images.
PSNR(img1, img2): Simplified PSNR for [0,1] range images.
calculate_ssim(img1, img2, border=0): SSIM for grayscale/color images.
ssim(img1, img2): Core SSIM with Gaussian filtering.

**Image I/O
**
load_img(filepath): Load RGB image.
save_img(filepath, img): Save RGB image.
load_gray_img(filepath): Load grayscale image.
save_gray_img(filepath, img): Save grayscale image.

**Visualization**

visualization(feature, save_path, type='max', colormap=cv2.COLORMAP_JET): Create heatmap from PyTorch feature map.

**Model Analysis
**
my_summary(test_model, H=256, W=256, C=3, N=1): Print model summary, FLOPs (GMac), and parameters (requires fvcore).

**Dependencies**

NumPy, OpenCV, Math, PDB.
fvcore (for FlopCountAnalysis): pip install fvcore.




## 2.  Contact
For support, email 20110720115@fudan.edu.cn


## 3. Citation
If this work is helpful to you, please cite it as：

M. Mahdizadeh, J. Cao, P. Ye and T. Chen, "Mamba-Based Progressive-Recovery Framework for Multimodal Low Light Image Enhancement," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2025.3623502.
keywords: {Lighting;Image enhancement;Image color analysis;Image restoration;Feature extraction;Estimation;Brightness;Image edge detection;Transformers;Image fusion;Multimodal low light enhancement;progressive-recovery framework;mamba-UNet;CNN-mamba},






```




