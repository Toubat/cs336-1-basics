from __future__ import annotations

from pathlib import Path

from cs336_basics.bpe.train import run_train_bpe
from cs336_basics.bpe.utils import bytes_to_unicode
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


def main() -> None:
    input_path = FIXTURES_PATH / "corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    # Load expected merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    print(f"actual merges count:   {len(merges)}")
    print(f"expected merges count: {len(reference_merges)}")

    # Find first differing index
    first_diff = None
    for i, (a, b) in enumerate(zip(merges, reference_merges)):
        if a != b:
            first_diff = i
            break
    if first_diff is None and len(merges) != len(reference_merges):
        first_diff = min(len(merges), len(reference_merges))

    print(f"first differing index: {first_diff}")
    if first_diff is not None:
        print("at index diff:")
        actual_item: tuple[bytes, bytes] | None = merges[first_diff] if first_diff < len(merges) else None
        expected_item: tuple[bytes, bytes] | None = (
            reference_merges[first_diff] if first_diff < len(reference_merges) else None
        )
        print("  actual  ", actual_item)
        print("  expected", expected_item)

    # Print FULL content for both lists
    print("\n===== FULL ACTUAL MERGES =====")
    for i, m in enumerate(merges):
        print(i, m)

    print("\n===== FULL EXPECTED MERGES =====")
    for i, m in enumerate(reference_merges):
        print(i, m)

    # Write to files for external diff (VS Code)
    actual_out = Path("debug_merges_actual.txt")
    expected_out = Path("debug_merges_expected.txt")
    with open(actual_out, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{bytes_to_unicode(left)} {bytes_to_unicode(right)}\n")
    with open(expected_out, "w", encoding="utf-8") as f:
        for left, right in reference_merges:
            f.write(f"{bytes_to_unicode(left)} {bytes_to_unicode(right)}\n")
    print(f"\nWrote files:\n  {actual_out.resolve()}\n  {expected_out.resolve()}")


if __name__ == "__main__":
    main()
