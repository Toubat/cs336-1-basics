import os

import numpy as np
from loguru import logger

from cs336_basics.bpe.tokenizer import encode_file_stream


def stream_tokens_to_npy(
    input_filepath: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list,
    output_path: str,
    estimated_tokens: int | None = None,  # optional estimate
    flush_every: int = 1_000_000,  # how often flush
    vocab_size: int = 10000,  # vocab size for validation
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if estimated_tokens is None:
        # if you don't know total size, you can start with a chunk and then append,
        # but `.npy` format doesn't support easy appending natively
        raise ValueError("Need estimate_tokens to allocate .npy upfront")

    # Pre-allocate a memmap (so you write into the .npy file directly)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(estimated_tokens,))

    pos = 0
    for token in encode_file_stream(input_filepath, vocab_path, merges_path, special_tokens):
        if not isinstance(token, int | np.integer):
            raise ValueError(
                f"Token {token} (type: {type(token)}) is not an integer - vocab keys may not be properly converted"
            )

        if not (0 <= token < vocab_size):
            raise ValueError(f"Token {token} is out of range [0, {vocab_size})")

        if pos >= estimated_tokens:
            logger.warning("Estimated tokens exceeded. Stopping encoding.")
            break

        arr[pos] = token
        pos += 1

        if pos % flush_every == 0:
            arr.flush()  # optional flush to disk periodically

    # If actual count < estimate → you could either leave the rest unused, or further trim.
    arr.flush()
    logger.info("Final position: {} tokens", pos)


if __name__ == "__main__":
    stream_tokens_to_npy(
        input_filepath="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_path="./tokenizers/tinystories_gpt4_train_vocab.json",
        merges_path="./tokenizers/tinystories_gpt4_train_merges.txt",
        special_tokens=["<|endoftext|>"],
        output_path="../../tinystories_gpt4_train.npy",
        estimated_tokens=550000000,
    )
