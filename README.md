---
license: mit
language:
- en
- zh
- ja
- ko
- ru
metrics:
- accuracy
pipeline_tag: question-answering
library_name: adapter-transformers
version: 0.5
base_model:
- google/vit-base-patch16-224
- openai/whisper-base
---

# Axonura X1

## Description

Axonura X1 is a language model built using the GPT-6 Like architecture. It is designed to understand and generate text based on the input it receives.

## Features

- GPT-2 Like architecture
- Pre-trained on a large corpus of text
- Easy to use API
- Fast and efficient inference

## Usage
Create A `.env` File With Your Weights & Biases Credentials:
```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=axonura-x1-training
WANDB_ENTITY=your_team_or_username
```
Python 3.9 Version Or Later Must Be Installed On Your Machine.
```bash
pip install -r requirements.txt
python3 build.py
python3 test.py
```
