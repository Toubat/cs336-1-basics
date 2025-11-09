import json
from pathlib import Path

from loguru import logger

from cs336_basics.bpe.train import run_train_bpe
from cs336_basics.bpe.utils import bytes_to_unicode

TOKENIZER_PATH = Path("tokenizers")
TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)


def save_tokenizer(
    path: Path,
    name: str,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
):
    with open(path / f"{name}_vocab.json", "w") as f:
        json.dump(
            {v: bytes_to_unicode(token_bytes) for v, token_bytes in vocab.items()},
            f,
            indent=2,
        )
    with open(path / f"{name}_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{bytes_to_unicode(merge[0])} {bytes_to_unicode(merge[1])}\n")


def main():
    logger.info("Training tinystories_gpt4_train BPE tokenizer")
    vocab, merges = run_train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )

    logger.info("Saving tinystories_gpt4_train BPE tokenizer")
    save_tokenizer(TOKENIZER_PATH, "tinystories_gpt4_train", vocab, merges)

    logger.info("Training owt_train BPE tokenizer")
    vocab, merges = run_train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )

    logger.info("Saving owt_train BPE tokenizer")
    save_tokenizer(TOKENIZER_PATH, "owt_train", vocab, merges)


if __name__ == "__main__":
    main()
