import json
from collections.abc import Iterable, Iterator

from cs336_basics.bpe.pretokenize import pretokenize_text_iter
from cs336_basics.bpe.utils import BytePair, TokenRef, split_bytes
from tests.common import gpt2_bytes_to_unicode


def _add_special_tokens(vocab: dict[int, bytes], special_tokens: list[str] | None = None):
    vocab_values = set(vocab.values())

    if special_tokens is None:
        return

    for special_token in special_tokens:
        byte_encoded_special_token = special_token.encode("utf-8")
        if byte_encoded_special_token in vocab_values:
            continue
        vocab[len(vocab)] = special_token.encode("utf-8")


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        _add_special_tokens(vocab, special_tokens)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_bytes_to_ids: dict[bytes, int] = {vocab_token: vocab_id for vocab_id, vocab_token in vocab.items()}
        self.special_token_bytes: list[bytes] = [
            special_token.encode("utf-8") for special_token in special_tokens or []
        ]

    @classmethod
    def from_file(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ):
        raw_vocab: dict[int, str] = {}
        raw_merges: list[tuple[str, str]] = []
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_path, "rb") as f:
            raw_vocab = json.load(f)

        with open(merges_path) as f:
            for line in f:
                token_bytes = line.rstrip().split(" ")
                assert len(token_bytes) == 2, f"Invalid merge line: {line}"
                raw_merges.append((token_bytes[0], token_bytes[1]))

        vocab = {
            vocab_id: bytes([gpt2_byte_decoder[b] for b in vocab_token]) for vocab_id, vocab_token in raw_vocab.items()
        }
        merges = [
            (
                bytes([gpt2_byte_decoder[b] for b in left]),
                bytes([gpt2_byte_decoder[b] for b in right]),
            )
            for left, right in raw_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretoken_to_ids: dict[bytes, list[int]] = {}
        token_ids: list[int] = []

        for pretoken in pretokenize_text_iter(text, self.special_tokens):
            for _ in self._encode_pretoken_iter(pretoken, pretoken_to_ids):
                pass

            token_ids.extend(pretoken_to_ids[pretoken])

        return token_ids

    def _encode_pretoken_iter(self, pretoken: bytes, cached_pretoken_to_ids: dict[bytes, list[int]]) -> Iterator[int]:
        # for special tokens, we can just add the token ID to the list and return
        if pretoken in self.special_token_bytes:
            cached_pretoken_to_ids[pretoken] = [self.token_bytes_to_ids[pretoken]]
            yield self.token_bytes_to_ids[pretoken]
            return

        # if the pretoken is not cached, merge the pretoken
        if pretoken not in cached_pretoken_to_ids:
            cached_pretoken_to_ids[pretoken] = []
            token_ref = TokenRef(tokens=split_bytes(pretoken))
            bp_counts = token_ref.bp_counts

            # iterate over merges to merge the pretoken
            for merge in self.merges:
                bp = BytePair(merge[0], merge[1])
                if bp not in bp_counts:
                    continue
                token_ref.merge(bp)
                bp_counts = token_ref.bp_counts

            cached_pretoken_to_ids[pretoken] = [self.token_bytes_to_ids[token] for token in token_ref.tokens]

        yield from cached_pretoken_to_ids[pretoken]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        pretoken_to_ids: dict[bytes, list[int]] = {}

        for text in iterable:
            for pretoken in pretokenize_text_iter(text, self.special_tokens):
                yield from self._encode_pretoken_iter(pretoken, pretoken_to_ids)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
