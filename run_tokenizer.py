from cs336_basics.bpe.tokenizer import encode_file_stream

if __name__ == "__main__":
    for token in encode_file_stream(
        "./data/owt_valid.txt",
        "./tokenizers/owt_valid_vocab.json",
        "./tokenizers/owt_valid_merges.txt",
        ["<|endoftext|>"],
    ):
        pass
