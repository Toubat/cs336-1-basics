from __future__ import annotations

import os
from collections.abc import Iterator
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from loguru import logger
from tqdm import tqdm

from cs336_basics.bpe.utils import PAT, find_chunk_boundaries


def cpu_count() -> int:
    return os.cpu_count() or 1


def pretokenize_text_iter(text: str, special_tokens: list[str]) -> Iterator[bytes]:
    if len(special_tokens) == 0:
        for pre_token in re.finditer(PAT, text):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes
        return

    start_index = 0
    # Sort special tokens by length in descending order to match longer tokens first
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    for match in re.finditer("|".join(re.escape(token) for token in sorted_tokens), text):
        matched_token_bytes = match.group().encode("utf-8")

        for pre_token in re.finditer(PAT, text, pos=start_index, endpos=match.start()):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes

        yield matched_token_bytes
        start_index = match.end()

    if start_index < len(text):
        for pre_token in re.finditer(PAT, text, pos=start_index, endpos=len(text)):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes


def pretokenize_text(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    pretoken_counts: dict[bytes, int] = {}

    # Sort special tokens by length in descending order to match longer tokens first
    sorted_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
    chunk_iter = (
        re.splititer("|".join(re.escape(token) for token in sorted_tokens), text)
        if len(sorted_tokens) > 0
        else [text]
    )

    for chunk in chunk_iter:
        for pre_token in re.finditer(PAT, chunk):
            pre_token_bytes = pre_token.group().encode("utf-8")

            if pre_token_bytes not in pretoken_counts:
                pretoken_counts[pre_token_bytes] = 0
            pretoken_counts[pre_token_bytes] += 1

    return pretoken_counts


def pretokenize_file_chunk(
    input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]
) -> dict[bytes, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    return pretokenize_text(text, special_tokens)


def merge_counts(a: dict[bytes, int], b: dict[bytes, int]) -> dict[bytes, int]:
    for key, value in b.items():
        if key not in a:
            a[key] = 0
        a[key] += value
    return a


def pretokenize_file(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[bytes, int]:
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
            results.append(pool.apply_async(pretokenize_file_chunk, (input_path, start, end, special_tokens)))

        for result in tqdm(results, desc="Parallel Pretokenization"):
            counts = result.get()
            merge_counts(pretoken_counts, counts)
    logger.debug("Pretokenization complete: {} unique pretoken strings", len(pretoken_counts))

    return pretoken_counts
