# TKC-AI

An experimental decoder-only language model developed from random initialization.
The repository does not use pretrained model weights.

## Current status

- [x] Decoder-only Transformer architecture
- [x] Reversible UTF-8 byte tokenizer (starter tokenizer)
- [x] Next-token training loop with validation loss, atomic checkpoints and resumable optimizer/RNG state
- [x] Response-only supervised instruction tuning
- [x] Small Vietnamese reasoning-example seed for pipeline testing
- [x] Staged assistant orchestration: plan, research, retrieve, draft, self-review, answer
- [x] Deterministic symbolic reasoning for arithmetic, word problems, linear equations, syllogisms, conditionals, sequences and weekday logic
- [x] Checkable derivation steps and checkpoint-free reasoning/evidence-only chat mode
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
  reasoning.py     # deterministic, checkable symbolic reasoning tools
  research.py      # optional live web-search adapter
  knowledge.py     # persistent retrieval memory and curated SFT examples
  teach.py         # save a fact or a reviewed question/answer example
  chat.py          # interactive chat; symbolic tools work even without a trained checkpoint
training/
  train.py         # pretraining, separate validation, atomic save and full-state resume
  sft.py           # response-only supervised instruction tuning; accepts many JSONL files
  evaluate.py      # exact-answer smoke evaluation on held-out model outputs
  evaluate_symbolic.py # score the deterministic rule-based reasoner
  build_reasoning_curriculum.py # generate synthetic, solver-verified SFT examples
tests/
  test_agent_memory.py
  test_generation.py
  test_sft_data.py
  test_evaluation.py
  test_reasoning.py
  test_agent_fallback.py
  test_reasoning_dataset.py
  test_reasoning_curriculum.py
  test_chat_teaching.py
  test_training_split.py
  test_training_resume.py
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

This creates checkpoints/tkc_ai_latest.pt. The last 5% of tokens are reserved for a separate validation partition by default. Training now stops with a clear error if either partition is too short for a sequence window; it never silently reuses training tokens as validation data. For a short smoke-test corpus, reduce --seq-len or increase --val-fraction (which must remain below 0.5), for example:

~~~bash
python -m training.train --steps 10 --seq-len 32 --batch-size 2 --val-fraction 0.2
~~~

Validation samples are repeatable and do not consume the random-number stream used for training. A small corpus can check the pipeline only; it will not teach broad language ability.

### Resume interrupted pretraining

New checkpoints save model weights, AdamW optimizer state, Python/PyTorch RNG states, model configuration, batch size, validation split and a SHA-256 fingerprint of the normalized training corpus. Checkpoints are written to a temporary file and atomically replaced to reduce the risk of a partially written checkpoint.

~~~bash
python -m training.train --resume checkpoints/tkc_ai_latest.pt --steps 2000
~~~

The steps argument is the total target step, not the number of additional steps. When resuming, saved context length, batch size, validation fraction and optimizer settings are retained unless the learning rate is explicitly overridden with lr. The corpus fingerprint is checked; a different corpus is rejected unless allow-data-change is explicitly supplied. Changing the corpus means the run is no longer directly comparable with the original run.

Older checkpoints created before optimizer/RNG metadata was added can still restore model weights, but the optimizer and random-number streams must start fresh. The code prints warnings when full resume metadata is missing. Reproducibility is intended for the same corpus and compatible software/device environment; bit-for-bit equality across different hardware or nondeterministic GPU kernels is not guaranteed.

## Stage 2: instruction tuning

After you have a base checkpoint from Stage 1, run:

~~~bash
python -m training.sft --checkpoint checkpoints/tkc_ai_latest.pt --data data/reasoning_examples.jsonl data/learned_examples.jsonl --steps 300
~~~

Pass only datasets that exist. The data argument accepts one or more JSONL files containing prompt and answer. The trainer masks prompt tokens and calculates loss only on answer tokens. Do not use the tiny seed as a substitute for a real training corpus.

## Teach TKC-AI

During chat, you can save a fact with `/teach Chủ đề | Nội dung đã kiểm chứng` or save a reviewed training pair with `/example Câu hỏi | Câu trả lời đã kiểm chứng`. The CLI commands below provide the same functionality.

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

Chat can also start before a trained checkpoint exists:

~~~bash
python -m brain.chat
~~~

Without a checkpoint, TKC-AI runs its deterministic solver for supported math and logic patterns, can retrieve taught notes, and can show web-search excerpts when search is requested. It explicitly refuses to pretend that it can generate a general answer from untrained weights. After a trained checkpoint exists, pass it with --checkpoint to enable the language-model planning, drafting and review pipeline.

With a trained checkpoint, the agent forms a short plan, searches the web when requested or when current information is needed, retrieves relevant taught notes, drafts an answer, runs a self-review pass, and then produces the final response. It prints any retrieved source URLs. The symbolic solver is used first for question patterns it can prove and verify.

For a single question:

~~~bash
python -m brain.chat --checkpoint checkpoints/tkc_ai_instruct.pt --question "Giải thích vì sao 17 + 28 = 45"
~~~

Web search is only one evidence source; verify important claims using primary sources. Search results are treated as untrusted data, not instructions.


## Build a verified reasoning curriculum

Create a reproducible set of synthetic arithmetic, word-problem, linear-equation, sequence, syllogism, conditional-logic and weekday examples. Answers and derivations are produced by the deterministic solver, not sampled from the untrained language model.

~~~bash
python -m training.build_reasoning_curriculum --count-per-type 100 --seed 42 --output data/reasoning_curriculum.jsonl
~~~

After a base checkpoint has been trained, this file can be included in supervised tuning:

~~~bash
python -m training.sft --checkpoint checkpoints/tkc_ai_latest.pt --data data/reasoning_examples.jsonl data/reasoning_curriculum.jsonl --steps 1000
~~~

Synthetic examples teach narrow task formats only. They are not a general knowledge corpus and must never be mixed into the held-out evaluation set.

## Independent reasoning smoke evaluation

The held-out evaluation examples are kept separate from training data. After a checkpoint exists, run:

~~~bash
python -m training.evaluate_symbolic --data data/evaluation/reasoning_smoke.jsonl
~~~

To evaluate the trained language model separately, run:

~~~bash
python -m training.evaluate --checkpoint checkpoints/tkc_ai_instruct.pt --data data/evaluation/reasoning_smoke.jsonl
~~~

This small exact-answer score measures the trained language model only; it is not a broad measure of intelligence or generalization. Keep future evaluation data separate from every training dataset. The checkpoint-free symbolic engine is separately checked against the same held-out examples by the unit test `test_reasoning_dataset.py`. You can also run its score report without any model checkpoint:

## Tests

Run the project's unit tests with:

~~~bash
python -m unittest discover -s tests
~~~

GitHub Actions is configured to run the unit tests on pushes and pull requests. Each push to main or pull request triggers a test run. A passing unit-test suite validates tested code paths, not training quality or broad reasoning ability. No end-to-end model-quality result is claimed.

The symbolic engine has real but narrow reasoning ability: it applies explicit rules and arithmetic, solves supported first-degree equations and returns checkable steps. It does not have consciousness or unrestricted human-like thought, and it cannot solve arbitrary language, factual, scientific or programming questions.

The Transformer language model still needs a large, legally usable corpus and substantial training before it can converse broadly. Its current self-review passes are generated by the same model and are not an independent verifier. Generated quality depends on data, context length, compute and evaluation. The starter byte tokenizer is inefficient and should later be replaced with a corpus-trained tokenizer.
