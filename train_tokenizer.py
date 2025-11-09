import json
from pathlib import Path

from loguru import logger

from cs336_basics.bpe.train import run_train_bpe
from cs336_basics.bpe.utils import bytes_to_unicode

TOKENIZER_PATH = Path("tokenizers")
TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)


TOKENIZE_DATASETS = {
    "tinystories_gpt4_valid": {
        "input_path": "data/TinyStoriesV2-GPT4-valid.txt",
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    },
    "owt_valid": {
        "input_path": "data/owt_valid.txt",
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    },
    "tinystories_gpt4_train": {
        "input_path": "data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    },
    "owt_train": {
        "input_path": "data/owt_train.txt",
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    },
}


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
    for name, dataset in TOKENIZE_DATASETS.items():
        logger.info(f"Training {name} BPE tokenizer")
        vocab, merges = run_train_bpe(
            input_path=dataset["input_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            verbose=True,
        )
        logger.info(f"Saving {name} BPE tokenizer")
        save_tokenizer(TOKENIZER_PATH, name, vocab, merges)


if __name__ == "__main__":
    main()
