# GKC-AI Roadmap

> Goal: build a useful Vietnamese-first AI assistant and Discord bot in measurable stages.
> Reality: this is a research project. A strong assistant requires a trained or hosted language model, reliable tools, evaluation, and safe deployment. Do not describe the current random-initialized Transformer as an intelligent chat model.

## Product target

The first useful release should:
- Hold natural Vietnamese conversations using a capable language model.
- Remember user-approved preferences and conversation facts with controls to inspect, correct, and delete them.
- Use tools for arithmetic, retrieval, and optional web research; cite sources and distinguish facts from guesses.
- Help with explanations and code while clearly stating uncertainty.
- Run as a Discord bot with per-server configuration, rate limits, privacy-conscious logging, and restart recovery.

## Phase 0 — Establish a reproducible baseline (first task)

- [ ] Run the complete unit-test suite and record the actual result.
- [ ] Run `python -m brain.doctor` and save a hardware/dependency readiness report.
- [ ] Audit chat, memory, model loading, and Discord integration points.
- [ ] Fix database concurrency: use SQLite WAL, bounded busy timeout, short transactions, and one connection per worker/thread where appropriate.
- [ ] Add a single documented configuration file and environment-variable handling; never commit secrets.
- [ ] Add CI checks for tests, formatting, and basic security checks.

**Exit criteria:** a clean install and repeatable test/doctor report; no claims of capabilities that were not tested.

## Phase 1 — Make GKC-AI genuinely conversational

- [ ] Choose the inference strategy based on the actual VPS: a small local quantized model if memory/CPU allow, or a reputable hosted model API if not.
- [ ] Keep the model provider behind a common interface so the provider can be changed without rewriting the bot.
- [ ] Add streaming responses, conversation context windows, graceful timeouts, retry/backoff, and clear error messages.
- [ ] Add Vietnamese-first system instructions, concise response defaults, and multilingual support.
- [ ] Add a small, human-reviewed Vietnamese evaluation set covering greetings, follow-up context, explanations, refusals, and hallucination checks.

**Exit criteria:** the assistant can answer ordinary Vietnamese conversation prompts coherently on a repeatable test set. Do not train the tiny random Transformer and expect it to become ChatGPT.

## Phase 2 — Reliable memory and retrieval

- [ ] Separate user profile memory, per-channel conversation history, and factual knowledge.
- [ ] Store only information the user is allowed to share; provide commands to inspect, correct, opt out, and delete memory.
- [ ] Add retrieval ranking, timestamps, source/provenance, and stale-fact handling.
- [ ] Prevent one Discord user's private memory from leaking to another user or server.
- [ ] Test concurrent writes and database recovery.

**Exit criteria:** memory survives restarts, respects user/server boundaries, and can be managed by users.

## Phase 3 — Reasoning and tools

- [ ] Keep deterministic math/logic solvers as tools, not as a claim of general intelligence.
- [ ] Add structured tool calling with schemas, argument validation, timeouts, and permission checks.
- [ ] Add calculator, safe code explanation, and optional web search with source links.
- [ ] Make the assistant show evidence for important factual claims and say when sources disagree.
- [ ] Add independent tests for tool selection, tool output validation, and multi-step tasks.

**Exit criteria:** tool-assisted tasks pass a held-out test suite; unsafe or unauthorized actions are blocked.

## Phase 4 — Discord bot MVP

- [ ] Create a Discord application and bot token outside the repository; store the token only in environment secrets.
- [ ] Implement slash commands such as `/chat`, `/help`, `/status`, `/memory`, and `/forget`.
- [ ] Support mention/reply-based conversations, per-channel settings, and conversation reset.
- [ ] Add per-user rate limits, message length limits, abuse protection, and graceful handling of Discord API limits.
- [ ] Add health checks, structured logs with sensitive content redacted, automatic restart, and service supervision.
- [ ] Document invite permissions using least privilege.

**Exit criteria:** bot can be invited to a test server, respond reliably, isolate users, and recover after restart.

## Phase 5 — Improve model quality responsibly

- [ ] Prefer a capable existing model initially; compare local and hosted options on the same Vietnamese evaluation set.
- [ ] Only fine-tune models with licenses that permit the intended use.
- [ ] Build a reviewed dataset from licensed/public-domain material and user-consented examples; track provenance and remove personal data.
- [ ] Split train/validation/test data before training and prevent duplicates/leakage.
- [ ] Evaluate factuality, Vietnamese fluency, reasoning, coding, latency, cost, and safety before each release.
- [ ] Consider training a small model from scratch only as a separate research experiment with realistic compute and data budgets.

**Exit criteria:** every model change has a reproducible evaluation report and a rollback path.

## Phase 6 — Operations and release

- [ ] Pin dependencies and create a reproducible deployment.
- [ ] Use least-privilege service accounts, firewall rules, secret rotation, and backups.
- [ ] Add uptime/latency/error monitoring and disk/memory alerts.
- [ ] Add scheduled database backups and test restore procedures.
- [ ] Publish setup, upgrade, troubleshooting, privacy, and moderation documentation.

**Exit criteria:** documented deployment, tested restore, and observable service health.

## First VPS checklist

When the VPS is actually provisioned, first inspect OS, CPU, RAM, free disk, GPU/driver availability, network, and permissions. Do not assume a GPU exists. With 16 GB RAM, 4 CPU cores, and 20 GB disk, start with a compact quantized model or a hosted model API; do not promise ChatGPT-class local training. Preserve disk for the OS, logs, model files, and backups.

## Definition of done for the first milestone

A user can open a Discord test server, mention GKC-AI in Vietnamese, have a coherent multi-turn conversation, receive a useful answer from the model, use math/search tools when appropriate, and have memory remain private and manageable. A test report documents what works and what does not.
