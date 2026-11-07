"""Inspect a local UTF-8 text corpus before training TKC-AI from scratch.

This tool does not download, rewrite, or upload data. It reports exact duplicate
files and whether the corpus is large enough for the configured train/validation
windows. Review licensing and quality of every source yourself.
"""
import argparse
import hashlib
from pathlib import Path


def inspect_corpus(folder, seq_len=256, val_fraction=0.05):
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be greater than 0 and less than 0.5.")

    paths = sorted(Path(folder).rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {folder}.")

    valid_files = []
    invalid_files = []
    empty_files = []
    hashes = {}
    total_bytes = 0
    total_chars = 0
    for path in paths:
        raw = path.read_bytes()
        if not raw.strip():
            empty_files.append(str(path))
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            invalid_files.append(str(path))
            continue
        digest = hashlib.sha256(raw).hexdigest()
        hashes.setdefault(digest, []).append(str(path))
        valid_files.append(str(path))
        total_bytes += len(raw)
        total_chars += len(decoded)

    duplicates = [files for files in hashes.values() if len(files) > 1]
    token_count = total_bytes + (2 if total_bytes else 0)
    train_tokens = int(token_count * (1.0 - val_fraction))
    validation_tokens = token_count - train_tokens
    minimum = seq_len + 2
    ready = (
        bool(valid_files)
        and train_tokens >= minimum
        and validation_tokens >= minimum
    )
    return {
        "files_found": len(paths),
        "valid_files": valid_files,
        "invalid_utf8_files": invalid_files,
        "empty_files": empty_files,
        "duplicate_groups": duplicates,
        "bytes": total_bytes,
        "characters": total_chars,
        "estimated_tokens": token_count,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "seq_len": seq_len,
        "minimum_tokens_per_partition": minimum,
        "ready_for_smoke_training": ready,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/training_data")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    try:
        report = inspect_corpus(args.data, args.seq_len, args.val_fraction)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Text files found: {report['files_found']}")
    print(f"Valid UTF-8 non-empty files: {len(report['valid_files'])}")
    print(f"Invalid UTF-8 files: {len(report['invalid_utf8_files'])}")
    print(f"Empty files: {len(report['empty_files'])}")
    print(f"Exact duplicate groups: {len(report['duplicate_groups'])}")
    print(f"UTF-8 bytes / estimated byte tokens: {report['bytes']:,}")
    print(f"Characters: {report['characters']:,}")
    print(
        f"Train / validation tokens: {report['train_tokens']:,} / "
        f"{report['validation_tokens']:,}"
    )
    print(f"Ready for smoke training: {'YES' if report['ready_for_smoke_training'] else 'NO'}")
    print("This is only a size/encoding check, not a data-quality or licensing approval.")
    if report["invalid_utf8_files"]:
        print("Files with invalid UTF-8:")
        for path in report["invalid_utf8_files"]:
            print(f"  - {path}")
    if report["empty_files"]:
        print("Empty files:")
        for path in report["empty_files"]:
            print(f"  - {path}")
    if report["duplicate_groups"]:
        print("Exact duplicate groups:")
        for group in report["duplicate_groups"]:
            print("  - " + " == ".join(group))
    if not report["ready_for_smoke_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
