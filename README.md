# SelectPrompt


![Teaser](./frame.png)


## Codebase Overview

```markdown
The repository is structured in a modular manner to promote clarity, reproducibility, and extensibility. 

.
├── clip/          # VLM backbone and related modules
├── configs/       # Experiment configurations for models and datasets
├── datasets/      # Data loading and preprocessing pipeline
├── output/        # Logs, checkpoints, and evaluation outputs
├── scripts/       # Shell scripts for running experiments
├── trainers/      # Core training framework and algorithm implementation
├── train.py       # Main script for training
└── utils.py       # Common utility functions

```

## Environments
```bash

cd Dassl.pytorch

# Create a conda environment
conda create -y -n selectprompt python=3.8

# Activate the environment
conda activate selectprompt

# Install torch and torchvision (please choose the appropriate version according to your device and system configuration)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install this library 
python setup.py develop

cd ..
```

## Datasets

The proposed method is evaluated on five widely used benchmark datasets covering fine-grained recognition, scene understanding, texture recognition, action classification, and real-world noisy-label learning. The dataset statistics are summarized below.

| Dataset | Classes | Train | Test | Objective |
|--------|--------:|------:|-----:|-----------|
| Caltech101 | 100 | 4,128 | 2,465 | Object recognition |
| Flowers102 | 102 | 4,093 | 2,463 | Flower classification |
| OxfordPets | 37 | 2,944 | 3,669 | Pet recognition |
| DTD | 47 | 2,820 | 1,692 | Texture classification |
| EuroSAT | 10 | 13,500 | 8,100 | Satellite scene classification |
| Food101N | 101 | 310,009 | 30,300 | Food category recognition |


## Label Noise Protocol

We consider two label noise settings, **symmetric noise** and **asymmetric noise**, with noise rates ranging from **12.5% to 75%**.

| Noise Setting | Corruption Mechanism | Evaluated Rates |
|--------------|----------------------|-----------------|
| Symmetric Noise | Uniform random label flipping across all non-ground-truth classes | 12.5%, 25%, 37.5%, 50%, 62.5%, 75% |
| Asymmetric Noise | Class-dependent flipping toward semantically related classes | 12.5%, 25%, 37.5%, 50%, 62.5%, 75% |


## How to Run

Before launching the experiments, please first edit the configuration variables in `run.sh`:

- `dataset`: dataset name  
- `n_shot`: number of shots per class  
- `seed`: random seed  

The script will automatically run experiments under both **symmetric** and **asymmetric** label noise settings, with noise rates ranging from **12.5%** to **75%**.

```bash
bash ./scripts/selectprompt/run.sh
