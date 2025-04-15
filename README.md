# LRS2-DM
The study focuses on addressing the challenge of small object detection in low-resolution remote sensing imagery. 

# 🚢 LRS2-DM: Small Ship Target Detection in Low-Resolution Remote Sensing Images Based on Diffusion Models

<p align="center">
  <img src="LRS2-DM.png" width="90%"/>
</p>

## 📌 Introduction

To mitigate resolution limitations, the method incorporates cognitive-conditioned input and a low-level super-resolution module to generate reference images as auxiliary guidance. This enhances the quality of the super-resolved outputs. Additionally, to compensate for the loss of fine details in small targets, a spatial refinement module is employed. This module sharpens object-level features and improves the accuracy of ship detection in remote sensing scenes.

## 🧠 Key Contributions

- **Cognitive-Conditioned Input (CCI):** Incorporates prior information (class and location embeddings) to enhance image-text alignment via ReMLP modules.
- **Low-Level Super-Resolution Module (L²SR):** Utilizes diffusion-based priors (e.g., DALL·E, VQ-VAE-2) to produce high-fidelity reference images for guiding super-resolution tasks.
- **Spatial Refinement Module (SRM):** Sharpens object-level features and enhances small object detection accuracy by refining spatial details.
- **Multi-Loss Design:** Introduces a joint loss for diffusion model training and detection head regularization (L<sub>DM</sub> + L<sub>reg</sub>).

## 📊 Result


<p align="center">
  <img src="detection.jpg" width="90%"/>
</p>

📭 Contact
If your have any comments or questions, feel free to contact chenyantong@dlmu.edu.cn.
