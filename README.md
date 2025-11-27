# Cinematic Colorizer (ColorFormer Implementation)

![Status](https://img.shields.io/badge/Status-Sprint%201-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Kornia](https://img.shields.io/badge/Kornia-Enabled-orange)

**Cinematic Colorizer** is a Deep Learning project aiming to colorize Black & White movies using Transfer Learning and GANs. This repository implements a high-performance, hybrid data pipeline capable of training on massive datasets.

## Architecture Overview

The project features a **Hybrid Data Pipeline** designed for flexibility and scale, now refactored into a modular architecture:

```text
cinematic-colorizer/
├── main.py               # Single entry point
├── configs/              # Configuration constants
└── src/
    ├── data/             # Data Loading Logic (Local & Remote)
    └── utils/            # Helper functions
```

- **Local Data**: Efficiently streams images from local `.tar` archives (e.g., MovieNet).
- **Remote Streaming**: Consumes datasets directly from Hugging Face (e.g., ImageNet, Places365) without local storage.
- **Unified Preprocessing**: Uses **Kornia** for GPU-accelerated color space conversion (`RGB` <-> `Lab`) and augmentations.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/cinematic-colorizer.git
    cd cinematic-colorizer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    Copy the example environment file and add your Hugging Face token (required for ImageNet).
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```ini
    HF_AUTH_TOKEN=hf_...
    ```

## Roadmap

- [x] **Sprint 1: Hybrid Data Pipeline**
    - [x] Local `.tar` streaming (MovieNet).
    - [x] Hugging Face Streaming (ImageNet, Places365).
    - [x] Kornia Preprocessing (RGB -> Lab, Normalization).
    - [x] Modular Refactoring.
- [ ] **Sprint 2: ColorFormer Model Architecture**
    - [ ] Implement Generator (Transformer/CNN hybrid).
    - [ ] Implement Discriminator.
- [ ] **Sprint 3: Training Loop**
    - [ ] GAN Loss functions.
    - [ ] Mixed Precision Training.
- [ ] **Sprint 4: Inference & UI**
    - [ ] Video processing pipeline.
    - [ ] Streamlit/Gradio Demo.

## Usage

**Visualize the Data Pipeline:**
```bash
# Visualize from Local MovieNet (auto-detected in dataset/MovieNet_valid/valid_mv_raw/)
python main.py --source local_movienet

# Visualize from Hugging Face Places365 (Streaming)
python main.py --source hf_places365 --hf_token $HF_AUTH_TOKEN
```
