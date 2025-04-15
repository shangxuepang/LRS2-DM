# LRS2-DM
The study focuses on addressing the challenge of small object detection in low-resolution remote sensing imagery. 

# 🚢 LRS2-DM: Small Ship Target Detection in Low-Resolution Remote Sensing Images Based on Diffusion Models

<p align="center">
  <img src="LRS2-DM.png." width="90%"/>
</p>

## 📌 Introduction

This project implements the core methodology from our research on small object detection in low-resolution remote sensing imagery. Our approach addresses the inherent challenges posed by limited resolution, particularly for tiny targets such as ships in complex scenes.

## 🧠 Key Contributions

- **Cognitive-Conditioned Input (CCI):** Incorporates prior information (class and location embeddings) to enhance image-text alignment via ReMLP modules.
- **Low-Level Super-Resolution Module (L²SR):** Utilizes diffusion-based priors (e.g., DALL·E, VQ-VAE-2) to produce high-fidelity reference images for guiding super-resolution tasks.
- **Spatial Refinement Module (SRM):** Sharpens object-level features and enhances small object detection accuracy by refining spatial details.
- **Multi-Loss Design:** Introduces a joint loss for diffusion model training and detection head regularization (L<sub>DM</sub> + L<sub>reg</sub>).

## 📊 Framework Overview

The architecture is divided into three main components:

1. **CCI:** Embeds class and location semantics to enhance the input conditions.
2. **L²SR:** Enhances image resolution with generative priors.
3. **SRM:** Refines spatial features for accurate object localization and classification.

<p align="center">
  <img src="LRS2-DM.png" width="90%"/>
</p>

