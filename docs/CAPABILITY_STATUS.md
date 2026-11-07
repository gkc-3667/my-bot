# TKC-AI capability status

This document separates shipped behavior from training-dependent goals.

## What can reason now, without a model checkpoint

The deterministic reasoner in \`brain/reasoning.py\` can return an answer and a checkable derivation for these limited patterns:

- Integer and decimal arithmetic with operator precedence, including word forms such as cộng, trừ, nhân and chia.
- Small multiplication and equal-sharing word problems.
- First-degree equations in one variable, followed by substitution of the solution to check both sides.
- Simple universal-category syllogisms when the premises and question match the supported form.
- Conditional reasoning that avoids affirming the consequent, and uses modus tollens when the consequent is explicitly negated in the observation statement.
- Arithmetic and geometric sequences, including negative arithmetic sequences. Continuation is marked lower-confidence because a finite sequence can fit more than one rule.
- Parity from a remainder modulo 2.
- Previous-day reasoning for a weekday question.

The chat interface can run this tool-only mode before a trained checkpoint exists. It can also retrieve facts explicitly saved in SQLite and display web-search snippets when the optional search dependency and network are available.

## What is not learned yet

The repository has no trained TKC-AI language-model checkpoint. The Transformer weights initialized from random values do not know Vietnamese, world facts, programming languages or general reasoning. The 12 starter examples and generated synthetic curriculum are narrow demonstrations, not general language pretraining.

Without trained weights, TKC-AI cannot reliably:
- Answer arbitrary factual or open-ended questions.
- Understand long natural-language context outside its explicit rule patterns.
- Write high-quality code or explain arbitrary subjects.
- Learn new capabilities from a fact stored in SQLite. That store is retrieval memory, not a weight update.
- Independently verify arbitrary web claims or generated code.

After pretraining and supervised fine-tuning, the language-model path can draft broader answers. Its self-review uses the same model and is not an independent guarantee of correctness. Code execution and secure sandboxing are not implemented.

## Meaning of “thinking”

There is no claim of consciousness or human-like subjective thought. The current system performs limited computational inference: it parses supported premises, applies explicit rules, calculates results, checks some solutions, and displays the derivation. That is real reasoning for a narrow class of tasks, not general intelligence.

## How capability is measured

- \`tests/test_reasoning.py\` tests individual rules and refusal on unsupported cases.
- \`tests/test_reasoning_dataset.py\` keeps the symbolic solver aligned with a small held-out task set.
- \`training/evaluate_symbolic.py\` prints a rule-engine score without needing model weights.
- \`training/evaluate.py\` separately measures model-generated answers after a checkpoint exists.
- GitHub Actions runs the unit-test suite on pushes and pull requests.

A perfect score on a small rule-specific set means only that those rules work. It does not demonstrate broad generalization.
