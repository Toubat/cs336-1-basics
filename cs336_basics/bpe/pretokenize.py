from __future__ import annotations

import os
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from loguru import logger
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
