# Training corpus

Place UTF-8 plain-text files (`.txt`) in this folder or its subfolders.

Use only text you own, have permission to use, or that is clearly licensed for
the intended training use. Keep private, personal, confidential, and secret
information out of the corpus. Do not commit large datasets or copyrighted
books/articles without permission.

The initial tokenizer maps UTF-8 bytes directly to token IDs. It is reversible
but inefficient for multilingual text; a trained BPE tokenizer is a planned
upgrade.

A tiny sample corpus is useful only for checking that the training code runs.
It will not produce a capable assistant. Strong language ability needs a very
large, diverse, carefully filtered corpus, substantial compute, and extensive
evaluation.
