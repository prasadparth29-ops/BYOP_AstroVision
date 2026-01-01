# BYOP_AstroVision

-->AstroVision is a deep learning model that denoises low-light images mostly taken form smartphones and cameras
-->The model is based on an **Astr U-Net architecture**, which is a modified U-Net using the **LeakyReLU activations**.
-->The best performance was achieved at **epoch 34**.

## Loss:
![Testing Loss Curve](images/loss.png)
## PSNR
![PSNR Curve](images/psnr.png)
## SSIM
![SSIM Curve](images/ssim.png)

## Example Outputs

**Only Denoising**
![output1](images/Denoisingq.png)

**Only Constellation Marking**
![output2](images/constellation3.png)

**Denoising and Constellations Mapped**
![output3](images/denoise+constellation1.png)
![output4](images/denoise+constellation2.png)
