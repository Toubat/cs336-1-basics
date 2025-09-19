import json
import os

from cs336_basics.bpe.train import run_train_bpe
from cs336_basics.bpe.utils import bytes_to_unicode


def main():
    vocab, merges = run_train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/owt_vocab.json", "w") as f:
        json.dump(
            {v: bytes_to_unicode(token_bytes) for v, token_bytes in vocab.items()},
            f,
            indent=2,
        )

    with open("outputs/owt_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{bytes_to_unicode(merge[0])} {bytes_to_unicode(merge[1])}\n")


if __name__ == "__main__":
    main()
