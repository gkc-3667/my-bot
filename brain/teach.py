"""Save a user-taught fact or curated Q&A example for TKC-AI."""
import argparse

from brain.knowledge import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(
        description="Store notes for retrieval or curated examples for later SFT."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fact", help="A fact or lesson to store in retrieval memory.")
    group.add_argument("--prompt", help="Question/prompt for a curated SFT example.")
    parser.add_argument("--answer", help="Verified answer; required together with --prompt.")
    parser.add_argument("--topic", default="", help="Short knowledge topic.")
    parser.add_argument("--source", default="", help="Where the fact was verified.")
    parser.add_argument("--database", default="data/knowledge.sqlite3")
    parser.add_argument("--examples", default="data/learned_examples.jsonl")
    args = parser.parse_args()

    knowledge = KnowledgeBase(args.database)
    if args.fact is not None:
        item_id = knowledge.add(args.fact, topic=args.topic, source=args.source)
        print(f"Saved knowledge note #{item_id}. This updates retrieval memory, not model weights.")
        return

    if args.answer is None:
        parser.error("--answer is required when --prompt is used.")
    added = knowledge.add_training_example(args.prompt, args.answer, args.examples)
    if added:
        print(f"Saved curated training example to {args.examples}.")
        print("The model learns this only after a later supervised fine-tuning run.")
    else:
        print("That prompt/answer example already exists; no duplicate was added.")


if __name__ == "__main__":
    main()
