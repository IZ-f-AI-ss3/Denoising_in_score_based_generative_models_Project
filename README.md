# Denoising in Score-Based Generative Models

This repository studies **sampling strategies for score-based generative models**, extending **Annealed Langevin Dynamics** (Song & Ermon, 2019) with a simple **Half-Denoising** (Hyvarinen, 2025) modification to reduce sampling bias while preserving good mode exploration.

We compare:
- **Standard Annealed Langevin Dynamics**
- **Annealed Half-Denoising (ours)**

Experiments are conducted on **MNIST**, **CELEBA** and **CIFAR-10** using pretrained NCSN checkpoints.  
This repository accompanies a course project for **Probabilistic Graphical Models (MVA, ENS Paris-Saclay)**.

---

## Setup

Clone the repository:
```bash
git clone https://github.com/IZ-f-AI-ss3/Denoising_in_score_based_generative_models.git
cd Denoising_in_score_based_generative_models
```

Install dependencies:
```bash
pip install torch torchvision
pip install pyyaml tqdm pillow tensorboardX seaborn gdown
```

## Pretrained Checkpoints
We use pretrained NCSN checkpoints from the original implementation.

Download and extract them to the repository root:
```
gdown 1BF2mwFv5IRCGaQbEWTbLlAOWEkNzMe5O
unzip run.zip
```

## Sampling

All sampling is run via main.py using AnnealRunner.

### Standard Annealed Langevin Dynamics
```python
python main.py \
  --runner AnnealRunner \
  --heavy_test \
  --doc cifar10 \
  --sampling_type annealed \
  -  -o samples/anneal
```

### Annealed Half-Denoising (Ours)
```python
python main.py \
  --runner AnnealRunner \
  --heavy_test \
  --doc cifar10 \
  --sampling_type half_denoising \
  --o samples
  --doc celeba
  --n_samples 11000
```

Both commands:

- Load the same pretrained model
- Differ only in the sampling procedure
- Save generated images to the specified output folder


## Results and Evaluation
Quantitative metrics (FID, Inception Score) and visual comparisons are computed using the notebooks in notebooks/, following the same workflow shown there:
- Generate samples
- Run the evaluation notebook
- Export figures and metrics

## Notes
Sampling is computationally expensive; generating 10k images may take several hours on a single GPU.
<br>
<br>

![Celeba](assets/celeba_large.gif)

