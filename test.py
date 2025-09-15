from cs336_basics.bpe.train import get_pretoken_counts

counts = get_pretoken_counts("data/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"])
