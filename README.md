# README

## Course Information
- **Program:** Specialization in Artificial Intelligence
- **SNIES Code:** 108149
- **University:** Universidad Nacional de Colombia
- **Faculty:** Facultad de Minas
- **Course Name:** Sistemas Inteligentes
- **Course Code (SIA):** 3008410

## Repository Purpose
This repository contains academic exercises, workshops, and a final project for the Intelligent Systems course. The work is practical and experimental, covering the full spectrum of modern AI: from classical discriminative models and generative image synthesis, through transformer architectures, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and AI Agents — with emphasis on experimentation, evaluation, and interpretation.

## Content Overview
- **Main topics:** discriminative vs. generative modeling, supervised classification, variational autoencoders, diffusion models, transformer self-attention, LLMs, fine-tuning and quantization, semantic similarity, emotion analysis, adversarial NLP attacks, model explainability, multimodal AI, RAG pipelines, and AI Agents.
- **Artifacts included:** Jupyter notebooks, datasets (CSV), generated images/videos, and a full-stack final project.
- **High-level structure:**
  - `week1/` — baseline classifiers, VAE latent space exploration, Stable Diffusion image generation
  - `week2/` — NLP with Transformers: translation, semantic search, fine-tuning, quantization, emotion analysis
  - `week4/` — multimodal AI with the Gemini API
  - `week6/` — adversarial text attacks and robustness evaluation
  - `week7/` — model explainability and mechanistic interpretability
  - `PROJECTS/` — final project: AI Labor Law Assistant

## Key Concepts Implemented
- Discriminative vs. generative modeling
- Supervised classification and evaluation (accuracy, F1, confusion matrices)
- Variational Autoencoders (VAEs) and latent space visualization
- Diffusion-based text-to-image generation and image editing
- Transformer self-attention mechanics and visualization
- Large Language Models (LLMs): fine-tuning, prompt engineering, and quantization
- Retrieval-Augmented Generation (RAG)
- AI Agents with tool use and intent routing
- Semantic similarity search with sentence embeddings
- Emotion and sentiment analysis with BERT-family models
- Adversarial NLP attacks and model robustness
- Mechanistic interpretability and logit lens analysis
- Multimodal AI (text + image) with the Gemini API

## Repository Analysis

### Packages and Libraries
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `torch` (PyTorch)
- `tensorflow`, `tf-keras`
- `transformers`, `datasets`, `evaluate`, `tokenizers`, `sentence-transformers`
- `diffusers`
- `huggingface_hub`
- `google-genai`
- `accelerate`, `safetensors`
- `textattack`, `textblob`, `nltk`, `rouge_score`
- `Pillow` (PIL), `imageio`
- `tqdm`

### Models Used
- **Stable Diffusion:** `stable-diffusion-v1-5`, `StableDiffusionInstructPix2PixPipeline`
- **T5 / Flan-T5:** `t5-small`, `google/flan-t5-small`
- **BERT variants:** `bert-base-uncased`, `roberta-base`, `dccuchile/bert-base-spanish-wwm-cased`
- **GPT-2** (mechanistic interpretability / logit lens)
- **DistilBERT**
- **Universal Sentence Encoder:** `google/universal-sentence-encoder/4`
- **Gemini:** `gemini-2.5-flash`, `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`
- **Classical ML:** `MultinomialNB`, `LogisticRegression` (scikit-learn)
- **Custom VAE** (PyTorch implementation)

### Techniques and Approaches
- Logistic regression and Multinomial Naive Bayes classification with stratified train/test splits
- VAE training and 2D latent space visualization
- Text-to-image generation and image-to-image editing with Stable Diffusion
- Self-attention visualization (scaled dot-product attention)
- Neural machine translation with T5 and BLEU evaluation
- Semantic similarity search via sentence embeddings
- LLM fine-tuning with ROUGE evaluation
- Post-training quantization for model compression
- Multi-class emotion classification with BERT-family models
- Adversarial text attacks (PWWS, TextFooler, DeepWordBug, BAE, HotFlip)
- Logit lens and layer-wise analysis for mechanistic interpretability
- Multimodal prompting with the Gemini API

### Methodologies
- Experimental evaluation using held-out test sets
- Comparative analysis between model families (discriminative vs. generative)
- Iterative prompt tuning and qualitative assessment for image generation
- Robustness evaluation under adversarial perturbation
- Interpretability-driven analysis of internal model representations

## Final Project

### AI Labor Law Assistant
> AI Labor Law Assistant is a RAG-powered chatbot for Colombian labor law that combines LangGraph-based intent routing, ChromaDB retrieval, and Groq/Gemini LLMs to deliver contextual legal answers with citations, while also handling general questions through a modern React + FastAPI architecture.

**Repository:** [evinracher/spe-ai-labor-law-assistant](https://github.com/evinracher/spe-ai-labor-law-assistant)

## Technologies and Tools
- **Languages:** Python, JavaScript (React)
- **Frameworks/Libraries:** PyTorch, TensorFlow, Diffusers, Hugging Face Transformers, LangGraph, ChromaDB, FastAPI, React
- **APIs:** Gemini API (Google), Groq API, Hugging Face Hub
- **Platforms/Tools:** Jupyter notebooks, local datasets, Google Colab

## How to Run / Reproduce
Exercises can be run locally with Jupyter or in Google Colab. Before running any notebook, make sure the required environment variables and API keys (e.g., `GOOGLE_API_KEY`, Hugging Face token) are properly set, as several exercises depend on external APIs.

## Skills Demonstrated
- Supervised and unsupervised learning model development and evaluation
- Generative modeling: VAEs and diffusion-based image synthesis
- Transformer architecture understanding and self-attention visualization
- LLM fine-tuning, quantization, and prompt engineering
- RAG pipeline design and implementation
- AI Agent development with tool use and intent routing
- Semantic similarity and embedding-based retrieval
- Adversarial robustness evaluation in NLP
- Model interpretability and mechanistic analysis
- Multimodal AI with vision-language models
- Full-stack AI application development (React + FastAPI)

## Academic Disclaimer
> **Disclaimer:**  
> Some code comments and variable names may appear in Spanish, as the course was taught in Spanish. The README and main documentation are provided in English for broader accessibility.

## Academic Context
This repository is part of a formal academic specialization program. The code prioritizes clarity, learning, and experimentation over production-level optimization.
