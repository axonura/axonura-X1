---
license: mit
language:
- en
metrics:
- accuracy
pipeline_tag: question-answering
library_name: adapter-transformers
version: 0.0.1
---

# Axonura X1

## Description

Axonura X1 is a language model built using the GPT-2 Like architecture. It is designed to understand and generate text based on the input it receives.

## Features

- GPT-2 Like architecture
- Pre-trained on a large corpus of text
- Easy to use API
- Fast and efficient inference

## Usage
Python 3.7 Version Or Later Must Be Installed On Your Machine.
```bash
python3 build.py
python3 test.py
```
Or You Can Use Dockerized Environment
```bash
docker build -t axonura-x1:latest .
docker run -it --rm axonura-x1 bash
```
If You Need GPU Access To Run The Command:
```bash
docker run --gpus all -it --rm axonura-x1 bash
```