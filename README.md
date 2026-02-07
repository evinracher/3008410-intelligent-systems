# README

## Course Information
- **Program:** Specialization in Artificial Intelligence
- **SNIES Code:** 108149
- **University:** Universidad Nacional de Colombia
- **Faculty:** Facultad de Minas
- **Course Name:** Sistemas Inteligentes
- **Course Code (SIA):** 3008410

## Repository Purpose
This repository contains academic exercises and workshops for the Intelligent Systems course. The work is primarily practical and experimental, focusing on supervised learning baselines and generative image modeling using Stable Diffusion, with emphasis on experimentation and interpretation.

## Content Overview
- Main topics: discriminative vs generative modeling, supervised classification, evaluation metrics, and text-to-image/image-to-image generation with diffusion models.
- Artifacts included: Jupyter notebooks, datasets, and generated images/videos.
- High-level structure: `Exercise 1` contains a classification notebook and a CSV dataset; `Exercise 3` contains a Stable Diffusion notebook plus supporting images and outputs.

## Key Concepts Implemented
- Discriminative vs generative modeling
- Supervised classification
- Train/test splitting and class distribution checks
- Model evaluation with accuracy, confusion matrices, and reports
- Diffusion-based text-to-image generation
- Image-to-image transformation (instruct-pix2pix)
- Prompt engineering for image generation
- Simple video generation from image sequences

## Repository Analysis
### Packages and Libraries
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`
- `diffusers`
- `transformers`
- `huggingface_hub`
- `Pillow` (PIL)
- `imageio`
- `json`

### Techniques and Approaches
- Logistic regression and Multinomial Naive Bayes classification
- Train/test split with stratification
- Confusion matrix visualization and classification reports
- Text-to-image generation with Stable Diffusion
- Image-to-image editing with InstructPix2Pix
- Prompt refinement for visual quality
- Frame interpolation and video export with `imageio`

### Methodologies
- Experimental evaluation using held-out test sets
- Comparative modeling between discriminative and generative baselines
- Iterative prompt tuning and qualitative assessment for image generation

## Technologies and Tools
- **Languages:** Python
- **Frameworks/Libraries:** Scikit-learn, PyTorch, Diffusers, Hugging Face Hub/Transformers
- **Platforms/Tools:** Jupyter notebooks, local datasets, generated media artifacts

## How to Run / Reproduce
1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn torch diffusers transformers huggingface_hub pillow imageio "imageio[ffmpeg]"
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open and run `SISTEMAS-INTELIGENTES/Exercise 1/Generative vs Discriminative.ipynb`.
5. For `SISTEMAS-INTELIGENTES/Exercise 3/Stable Diffusion.ipynb`, create `AUTH.json` in the notebook folder with your Hugging Face token:
   ```json
   {"API_KEY":"YOUR_HF_TOKEN"}
   ```
6. Run the notebook cells in order to reproduce image generation and video creation outputs.

## Skills Demonstrated
- Supervised learning model development and evaluation
- Data handling and exploratory checks for class balance
- Comparative analysis between model families
- Diffusion-based generative modeling workflows
- Prompt engineering for image synthesis
- Reproducible experimentation in notebooks

## Academic Disclaimer
> **Disclaimer:**  
> Some code comments and variable names may appear in Spanish, as the course was taught in Spanish. The README and main documentation are provided in English for broader accessibility.

## Academic Context
This repository is part of a formal academic specialization program. The code prioritizes clarity, learning, and experimentation over production-level optimization.
