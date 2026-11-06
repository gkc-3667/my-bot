# TKC-AI

An experimental decoder-only language model developed from random initialization.
The repository does not use pretrained model weights.

## Current status

- [x] Decoder-only Transformer architecture
- [x] Reversible UTF-8 byte tokenizer (starter tokenizer)
- [x] Next-token training loop with validation loss and checkpoints
- [x] Basic autoregressive text generation from a trained checkpoint
- [ ] Train a more efficient BPE tokenizer from our own corpus
- [ ] Build a large, licensed, high-quality multilingual corpus
- [ ] Add instruction tuning and carefully designed reasoning examples
- [ ] Add robust evaluation suites for knowledge, logic, math, coding and safety
- [ ] Add memory, tools and planning components
- [ ] Serve the model through an API
- [ ] Connect a Discord bot

## Important reality check

The code is a research prototype, **not a smart assistant yet**. Randomly initialized weights do not contain language knowledge or reasoning ability. Training on a tiny sample only verifies that the pipeline runs; it will not create ChatGPT-level capability.

A capable foundation model requires a very large and diverse legally usable dataset, substantial accelerator compute, careful optimization, repeated evaluation, and later instruction tuning. No model can reliably answer every possible question.

## Architecture

```text
brain/
  transformer.py   # decoder-only Transformer
  tokenizer.py     # UTF-8 byte tokenizer (IDs 0..255, special IDs 256..258)
  generate.py      # checkpoint loading and autoregressive generation
training/
  train.py         # next-token training, validation and checkpointing
data/training_data/
  README.md        # corpus rules
checkpoints/
  .gitkeep
```

PyTorch is used as the tensor/autograd framework. The architecture and model weights are created by this project; no pretrained checkpoint is loaded.

## Setup

Use Python 3.10+ and install a PyTorch build appropriate for your CPU/GPU from https://pytorch.org/get-started/locally/.

```bash
python -m pip install -r requirements.txt
python brain/transformer.py
```

Add UTF-8 `.txt` files that you own or are licensed to use under `data/training_data/`.

## Train a small smoke test

The starter model is intentionally modest. A tiny corpus is only for checking the training pipeline:

```bash
python -m training.train --steps 100 --seq-len 128 --batch-size 2
```

For a longer run, adjust steps, sequence length and batch size to available hardware and corpus size. Training saves `checkpoints/tkc_ai_latest.pt`.

## Generate text

After a checkpoint exists:

```bash
python -m brain.generate --prompt "Xin chào" --max-new-tokens 200
```

Generated text quality depends entirely on training data, compute and training quality. The starter byte tokenizer is inefficient and is intended to be replaced by a corpus-trained BPE tokenizer.
