# TKC-AI

An experimental decoder-only language model developed from random initialization.
The repository does not use pretrained model weights.

## Current status

- [x] Decoder-only Transformer architecture
- [x] Reversible UTF-8 byte tokenizer (starter tokenizer)
- [x] Next-token training loop with validation loss and checkpoints
- [x] Response-only supervised instruction tuning
- [x] Small Vietnamese reasoning-example seed for pipeline testing
- [x] Staged assistant orchestration: plan, research, retrieve, draft, self-review, answer
- [x] Optional live web search through the ddgs search library
- [x] Persistent user-taught knowledge memory (SQLite)
- [x] Curated Q&A examples saved for later supervised fine-tuning
- [x] Code-writing and tutoring response modes
- [x] Interactive chat entry point and EOS-aware generation
- [x] Held-out reasoning smoke set and an exact-answer evaluation script
- [x] GitHub Actions workflow for running unit tests
- [ ] Train a more efficient BPE tokenizer from our own corpus
- [ ] Build a large, licensed, high-quality multilingual corpus
- [ ] Expand verified instruction/reasoning/code data substantially
- [ ] Add independent, broader evaluation suites for knowledge, logic, math, coding and safety
- [ ] Improve context length and model efficiency
- [ ] Serve the model through a secured API
- [ ] Connect a Discord bot

## Important reality check

This is still a research prototype, **not a smart assistant yet**. The repository currently has no trained checkpoint. Randomly initialized weights contain no learned language knowledge or reasoning ability. The included 12 reasoning examples are only a pipeline smoke-test seed.

The staged agent does not magically create intelligence. Its planning, draft and self-review passes all use the same TKC-AI checkpoint, so the reviewer can repeat the same mistake. The code-writing mode generates code but does not execute it. Web search can retrieve current references, but search snippets can be incomplete or wrong and must be checked.

A capable model needs a large, diverse, legally usable dataset, substantial accelerator compute, careful optimization, instruction tuning, and repeated independent evaluation. No model can reliably answer every possible question.

## Architecture

~~~text
brain/
  transformer.py   # decoder-only Transformer trained from random initialization
  tokenizer.py     # UTF-8 byte tokenizer (IDs 0..255, special IDs 256..258)
  generate.py      # safe checkpoint loading and EOS-aware generation
  agent.py         # staged plan -> web -> memory -> draft -> review -> answer
  research.py      # optional live web-search adapter
  knowledge.py     # persistent retrieval memory and curated SFT examples
  teach.py         # save a fact or a reviewed question/answer example
  chat.py          # interactive assistant entry point
training/
  train.py         # next-token pretraining, validation and checkpointing
  sft.py           # response-only supervised instruction tuning; accepts many JSONL files
  evaluate.py      # exact-answer smoke evaluation on held-out examples
tests/
  test_agent_memory.py
  test_generation.py
  test_sft_data.py
  test_evaluation.py
data/
  reasoning_examples.jsonl
  evaluation/reasoning_smoke.jsonl
  learned_examples.jsonl  # created when reviewed Q&A examples are taught
  knowledge.sqlite3       # created when facts are taught
.github/workflows/
  tests.yml               # run unit tests on pushes and pull requests
checkpoints/
  .gitkeep
~~~

PyTorch is used as the tensor/autograd framework. The architecture and model weights are created by this project; no pretrained checkpoint is downloaded. The instruction-tuning stage starts from a TKC-AI checkpoint produced by our own pretraining.

## Setup

Use Python 3.10+ and install a PyTorch build appropriate for your CPU/GPU from https://pytorch.org/get-started/locally/. The requirements file also installs ddgs for web search.

~~~bash
python -m pip install -r requirements.txt
python brain/transformer.py
~~~

Add UTF-8 .txt files that you own or are licensed to use under data/training_data/.

## Stage 1: next-token pretraining

~~~bash
python -m training.train --steps 100 --seq-len 128 --batch-size 2
~~~

This creates checkpoints/tkc_ai_latest.pt. A tiny corpus only checks that training runs; it does not teach broad language ability.

## Stage 2: instruction tuning

After you have a base checkpoint from Stage 1, run:

~~~bash
python -m training.sft --checkpoint checkpoints/tkc_ai_latest.pt --data data/reasoning_examples.jsonl data/learned_examples.jsonl --steps 300
~~~

Pass only datasets that exist. The --data argument accepts one or more JSONL files containing prompt and answer. The trainer masks prompt tokens and calculates loss only on answer tokens. Do not use the tiny seed as a substitute for a real training corpus.

## Teach TKC-AI

A fact or lesson can be saved to retrieval memory:

~~~bash
python -m brain.teach --fact "Python dùng thụt lề để xác định khối lệnh." --topic Python --source "Tài liệu đã kiểm chứng"
~~~

A reviewed question/answer pair can be saved to the next SFT dataset:

~~~bash
python -m brain.teach --prompt "Python dùng gì để xác định khối lệnh?" --answer "Python dùng thụt lề." --topic Python
~~~

Important: a saved fact is retrieved as external memory; it does not change model weights. A saved Q&A example changes model behavior only after it is included in a later SFT run. Review answers and sources before teaching them.

## Chat with the staged agent

After a trained checkpoint exists:

~~~bash
python -m brain.chat --checkpoint checkpoints/tkc_ai_instruct.pt
~~~

The agent forms a short plan first, searches the web when the request is time-sensitive or explicitly asks for search, retrieves relevant taught notes, drafts an answer, runs a self-review pass, and then produces the final response. It prints any retrieved source URLs.

For a single question:

~~~bash
python -m brain.chat --checkpoint checkpoints/tkc_ai_instruct.pt --question "Giải thích vì sao 17 + 28 = 45"
~~~

Web search is only one evidence source; verify important claims using primary sources. Search results are treated as untrusted data, not instructions.

## Independent reasoning smoke evaluation

The held-out evaluation examples are kept separate from training data. After a checkpoint exists, run:

~~~bash
python -m training.evaluate --checkpoint checkpoints/tkc_ai_instruct.pt --data data/evaluation/reasoning_smoke.jsonl
~~~

This small exact-answer score helps catch regressions; it is not a broad measure of intelligence or generalization. Keep future evaluation data separate from every training dataset.

## Tests

Run the project's unit tests with:

~~~bash
python -m unittest discover -s tests
~~~

GitHub Actions is configured to run the unit tests on pushes and pull requests. The tests are included in the repository, but this commit does not claim that they have passed in a target Python/PyTorch environment. No training run or end-to-end chat-quality evaluation is claimed.

Generated quality depends on the training corpus, context length, compute, and evaluation. The starter byte tokenizer is inefficient and should later be replaced with a corpus-trained tokenizer.
