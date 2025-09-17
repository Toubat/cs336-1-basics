from __future__ import annotations

import os
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from tqdm import tqdm

from cs336_basics.bpe.utils import PAT, BytePair, TokenRef, find_chunk_boundaries, split_bytes


def cpu_count() -> int:
    return os.cpu_count() or 1


def pretokenize(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> dict[bytes, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    pretoken_counts: dict[bytes, int] = {}
    for chunk in re.splititer("|".join(re.escape(token) for token in special_tokens), text):
        for pre_token in re.finditer(PAT, chunk):
            pre_token_bytes = pre_token.group().encode("utf-8")

            if pre_token_bytes not in pretoken_counts:
                pretoken_counts[pre_token_bytes] = 0
            pretoken_counts[pre_token_bytes] += 1

    return pretoken_counts


def merge_counts(a: dict[bytes, int], b: dict[bytes, int]) -> dict[bytes, int]:
    for key, value in b.items():
        if key not in a:
            a[key] = 0
        a[key] += value
    return a


def get_pretoken_counts(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[bytes, int]:
    parellel_count = cpu_count() * 2
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, parellel_count, b"<|endoftext|>")
        print("Finished finding chunk boundaries")

    pretoken_counts: dict[bytes, int] = {}
    with Pool(parellel_count) as pool:
        results: list[ApplyResult] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            results.append(pool.apply_async(pretokenize, (input_path, start, end, special_tokens)))

        for result in tqdm(results, desc="Parallel Pretokenization"):
            counts = result.get()
            merge_counts(pretoken_counts, counts)

    return pretoken_counts


def get_highest_bp(bp_to_counts: dict[BytePair, int]):
    max_bp, max_count = next(iter(bp_to_counts.items()))

    for bp, count in bp_to_counts.items():
        if count > max_count or (count == max_count and bp.merged_bytes > max_bp.merged_bytes):
            max_bp, max_count = bp, count

    return max_bp


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    pretoken_counts = get_pretoken_counts(input_path, special_tokens)
    token_refs = [TokenRef(tokens=split_bytes(pretoken), count=count) for pretoken, count in pretoken_counts.items()]

    bp_to_counts: dict[BytePair, int] = {}
    bp_to_token_ref_ids: dict[BytePair, set[int]] = {}

    for idx, token_ref in enumerate(token_refs):
        for bp, count in token_ref.bp_counts.items():
            bp_to_counts.setdefault(bp, 0)
            bp_to_token_ref_ids.setdefault(bp, set())

            bp_to_counts[bp] += count
            bp_to_token_ref_ids[bp].add(idx)

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {
        idx: b
        for idx, b in enumerate(
            [*[token.encode("utf-8") for token in special_tokens], *[bytes([i]) for i in range(256)]]
        )
    }

    while len(vocab) < vocab_size:
        bp = get_highest_bp(bp_to_counts)
        vocab[len(vocab)] = bp.merged_bytes

        for ref_idx in bp_to_token_ref_ids[bp]:
            token_ref = token_refs[ref_idx]

            curr_bp_counts = token_ref.bp_counts
            token_ref.merge(bp)
            next_bp_counts = token_ref.bp_counts

            for token_bp in curr_bp_counts:
                if token_bp not in next_bp_counts:
                    bp_to_token_ref_ids[token_bp].remove(ref_idx)

                curr_count = curr_bp_counts[token_bp]
                next_count = next_bp_counts[token_bp] if token_bp in next_bp_counts else 0
                bp_to_counts[token_bp] += next_count - curr_count

                assert bp_to_counts[token_bp] >= 0, (
                    f"Byte pair count cannot be negative: {token_bp}, got {bp_to_counts[token_bp]}"
                )

            for token_bp in next_bp_counts:
                # skip every processed byte pair from previous iteration
                if token_bp in curr_bp_counts:
                    continue

                # for each new byte pair, we update global map
                bp_to_counts.setdefault(token_bp, 0)
                bp_to_counts[token_bp] += next_bp_counts[token_bp]

                bp_to_token_ref_ids.setdefault(token_bp, set())
                bp_to_token_ref_ids[token_bp].add(ref_idx)

        bp_count, bp_ref_count = bp_to_counts[bp], len(bp_to_token_ref_ids[bp])
        assert bp_count == bp_ref_count == 0, (
            f"Byte pair count and reference count must be 0: {bp}, got {bp_count} and {bp_ref_count}"
        )
        del bp_to_counts[bp]
        del bp_to_token_ref_ids[bp]

    return vocab, merges


"""
# Got from pretokenization
list token_refs (list[TokenRef]) token_refs

# Compute below 2 data structures from token_refs
- dict byte_pair BytePair -> total_count (int) bp_to_counts
- dict byte_pair BytePair -> token_ref_ids (set[int]) bp_to_token_ref_ids

while less than vocab_size:
    bp = BytePair with highest count
    add it to the vocabulary

    token_refs_with_bp = find list of token refs contain this bp (use bp_to_token_ref_ids and token_refs)

    for each token_ref in token_refs_with_bp:
        compute curr_bp_to_counts for this token_ref
        perform merging on pretokenized text chunk
        compute next_bp_to_counts for this token_ref

        for each bp in curr_bp_to_counts:
            if bp not in next_bp_to_counts:
                remove token_ref id from bp_to_token_ref_ids[bp]

            curr_count = curr_bp_to_counts[bp]
            next_count = 0 if bp not in next_bp_to_counts else next_bp_to_counts[bp]
            bp_to_counts[bp] += next_count - curr_count
            !assert bp_to_counts[bp] never be negative
        
    bp_count, bp_ref_count = bp_to_counts[bp], len(bp_to_token_ref_ids[bp])
    !assert bp_count == bp_ref_count == 0

    remove bp from bp_to_counts
    remove bp from bp_to_token_ref_ids

"""
