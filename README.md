# TKC-AI

An experimental decoder-only language model developed from random initialization.
The repository does not use pretrained model weights.

## Current status

- [x] Decoder-only Transformer architecture
- [x] Reversible UTF-8 byte tokenizer (starter tokenizer)
- [x] Next-token training loop with validation loss and checkpoints
- [x] Basic autoregressive text generation from a trained checkpoint
- [x] Response-only supervised instruction-tuning script
- [x] Small Vietnamese reasoning-example seed for pipeline testing
- [ ] Train a more efficient BPE tokenizer from our own corpus
- [ ] Build a large, licensed, high-quality multilingual corpus
- [ ] Expand verified instruction/reasoning data substantially
- [ ] Add independent evaluation suites for knowledge, logic, math, coding and safety
- [ ] Add memory, tools and planning components
- [ ] Serve the model through an API
- [ ] Connect a Discord bot

## Important reality check

This is a research prototype, **not a smart assistant yet**. Randomly initialized weights contain no learned language knowledge or reasoning ability. The small example file is only enough to test the instruction-tuning pipeline; it will not make the model broadly intelligent. A capable model needs a large, diverse, legally usable dataset, substantial accelerator compute, careful optimization, repeated independent evaluation, and instruction tuning. No model can reliably answer every possible question.

## Architecture

```text
brain/
  transformer.py   # decoder-only Transformer
  tokenizer.py     # UTF-8 byte tokenizer (IDs 0..255, special IDs 256..258)
  generate.py      # checkpoint loading and autoregressive generation
training/
  train.py         # next-token pretraining, validation and checkpointing
  sft.py           # response-only supervised instruction tuning
data/
  reasoning_examples.jsonl  # small Vietnamese pipeline-test examples
checkpoints/
  .gitkeep
```

PyTorch is used as the tensor/autograd framework. The architecture and model weights are created by this project; no pretrained checkpoint is downloaded. The instruction-tuning stage starts from a TKC-AI checkpoint produced by our own pretraining.

## Setup

Use Python 3.10+ and install a PyTorch build appropriate for your CPU/GPU from https://pytorch.org/get-started/locally/.

```bash
python -m pip install -r requirements.txt
python brain/transformer.py
```

Add UTF-8 `.txt` files that you own or are licensed to use under `data/training_data/`.

## Stage 1: next-token pretraining

```bash
python -m training.train --steps 100 --seq-len 128 --batch-size 2
```

This creates `checkpoints/tkc_ai_latest.pt`. A tiny corpus only checks that training runs.

## Stage 2: instruction tuning

After you have a base checkpoint from Stage 1, run:

```bash
python -m training.sft --checkpoint checkpoints/tkc_ai_latest.pt --data data/reasoning_examples.jsonl --steps 300
```

This saves `checkpoints/tkc_ai_instruct.pt`. The trainer masks prompt tokens and calculates loss only on answer tokens. The included examples are a tiny pipeline test, not sufficient training data for a useful assistant.

## Generate text

```bash
python -m brain.generate --checkpoint checkpoints/tkc_ai_instruct.pt --prompt "Giải thích vì sao 17 + 28 = 45" --max-new-tokens 200
```

Generated quality depends on data, compute, and training. The starter byte tokenizer is inefficient and should later be replaced with a corpus-trained BPE tokenizer.
