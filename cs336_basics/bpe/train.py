from __future__ import annotations

import os
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from loguru import logger
from tqdm import tqdm

from cs336_basics.bpe.utils import PAT, BytePair, TokenRef, find_chunk_boundaries, split_bytes

# Disable library logs by default; can be enabled per-call with verbose=True
logger.disable("cs336_basics")


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
    logger.debug(
        "Starting pretokenization: input_path='{}', special_tokens={}, workers={}",
        input_path,
        special_tokens,
        parellel_count,
    )
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, parellel_count, b"<|endoftext|>")
        logger.debug(
            "Found {} chunk boundaries ({} chunks)",
            len(boundaries),
            max(0, len(boundaries) - 1),
        )

    pretoken_counts: dict[bytes, int] = {}
    with Pool(parellel_count) as pool:
        results: list[ApplyResult] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            results.append(pool.apply_async(pretokenize, (input_path, start, end, special_tokens)))

        for result in tqdm(results, desc="Parallel Pretokenization"):
            counts = result.get()
            merge_counts(pretoken_counts, counts)
    logger.debug("Pretokenization complete: {} unique pretoken strings", len(pretoken_counts))

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

    verbose: bool = bool(kwargs.get("verbose", False))
    if verbose:
        logger.enable("cs336_basics")

    logger.debug(
        "Begin BPE training: target_vocab_size={}, special_tokens_count={}",
        vocab_size,
        len(special_tokens),
    )
    pretoken_counts = get_pretoken_counts(input_path, special_tokens)
    token_refs = [TokenRef(tokens=split_bytes(pretoken), count=count) for pretoken, count in pretoken_counts.items()]
    logger.debug("Initialized {} token references", len(token_refs))

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

    logger.debug("Seeded base vocabulary with {} items", len(vocab))

    iteration = 0
    while len(vocab) < vocab_size and len(bp_to_counts) > 0:
        bp = get_highest_bp(bp_to_counts)
        new_token_id = len(vocab)
        vocab[new_token_id] = bp.merged_bytes
        merges.append((bp.left, bp.right))

        iteration += 1
        if iteration % 100 == 0 or iteration == 1:
            logger.debug(
                "Iteration {}: merged {} -> id {} (vocab={}/{})",
                iteration,
                bp,
                new_token_id,
                len(vocab),
                vocab_size,
            )

        bp_token_to_remove: dict[BytePair, set[int]] = {}
        for ref_idx in bp_to_token_ref_ids[bp]:
            token_ref = token_refs[ref_idx]

            curr_bp_counts = token_ref.bp_counts
            token_ref.merge(bp)
            next_bp_counts = token_ref.bp_counts

            for token_bp in curr_bp_counts:
                # we do not mutate bp_to_token_ref_ids here, since it will change set size during iteration
                if token_bp not in next_bp_counts:
                    bp_token_to_remove.setdefault(token_bp, set())
                    bp_token_to_remove[token_bp].add(ref_idx)

                curr_count = curr_bp_counts[token_bp]
                next_count = next_bp_counts[token_bp] if token_bp in next_bp_counts else 0
                bp_to_counts[token_bp] += next_count - curr_count

                assert bp_to_counts[token_bp] >= 0, (
                    f"Byte pair count cannot be negative: {token_bp}, got {bp_to_counts[token_bp]}"
                )

            for token_bp in next_bp_counts:
                # skip every processed byte pair from previous iteration, since we already processed it
                if token_bp in curr_bp_counts:
                    continue

                # for each new byte pair, we update global map
                bp_to_counts.setdefault(token_bp, 0)
                bp_to_counts[token_bp] += next_bp_counts[token_bp]

                bp_to_token_ref_ids.setdefault(token_bp, set())
                bp_to_token_ref_ids[token_bp].add(ref_idx)

        # remove token refs that are no longer needed, since they are no longer in the token ref
        for bp_to_remove, token_ref_ids in bp_token_to_remove.items():
            bp_to_token_ref_ids[bp_to_remove] -= token_ref_ids

        bp_count, bp_ref_count = bp_to_counts[bp], len(bp_to_token_ref_ids[bp])
        assert bp_count == bp_ref_count == 0, (
            f"Byte pair count and reference count must be 0: {bp}, got {bp_count} and {bp_ref_count}"
        )
        del bp_to_counts[bp]
        del bp_to_token_ref_ids[bp]

    logger.debug("BPE training complete: final_vocab_size={}, merges={}", len(vocab), len(merges))
    if verbose:
        logger.disable("cs336_basics")
    return vocab, merges
