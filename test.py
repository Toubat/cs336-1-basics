import json

from cs336_basics.bpe.train import run_train_bpe
from cs336_basics.bpe.utils import bytes_to_unicode


def main():
    vocab, merges = run_train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    with open("test_vocab.json", "w") as f:
        json.dump({v: str(b) for v, b in vocab.items()}, f, indent=2)

    with open("test_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{bytes_to_unicode(merge[0])} {bytes_to_unicode(merge[1])}\n")


if __name__ == "__main__":
    main()
