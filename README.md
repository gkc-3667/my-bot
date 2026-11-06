# TKC-AI

An experimental language-model project built from scratch.

## Current status

- [x] Initial decoder-only Transformer architecture
- [ ] Train a tokenizer from our own text corpus
- [ ] Prepare and validate training datasets
- [ ] Implement next-token training and checkpoints
- [ ] Evaluate language modeling and reasoning capabilities
- [ ] Add memory and planning components
- [ ] Serve the model through an API
- [ ] Connect a Discord bot

## Important

The current Transformer starts with randomly initialized weights. It is **not a trained chatbot** and cannot answer natural-language questions yet. It must first be trained on a sufficiently large, legally usable dataset and evaluated.

This project does not import pretrained model weights. PyTorch is used as the tensor/autograd framework; the model architecture and its weights are developed in this repository.

## Project structure

```text
brain/
  __init__.py
  transformer.py
requirements.txt
```

## Development environment

Python 3.10+ is recommended.

Install PyTorch using the command recommended for your machine at:
https://pytorch.org/get-started/locally/

Then install the project requirements:

```bash
python -m pip install -r requirements.txt
```

Run the architecture smoke test:

```bash
python brain/transformer.py
```

The first prototype uses a small configuration (vocabulary 8,000; context 512; width 256; 8 attention heads; 6 layers). This is an architecture test, not a claim of ChatGPT-level capability. Model size and training plan must be selected after checking the available hardware and data.
