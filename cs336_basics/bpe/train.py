import os
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from tqdm import tqdm

from cs336_basics.bpe.utils import PAT, find_chunk_boundaries


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

    return {}, []
