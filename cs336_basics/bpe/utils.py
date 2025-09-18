from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import BinaryIO, NamedTuple

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


BYTES_TO_UNICODE = gpt2_bytes_to_unicode()


class BytePair(NamedTuple):  # noqa: F821
    left: bytes
    right: bytes

    @property
    def merged_bytes(self):
        return self.left + self.right

    def __str__(self) -> str:
        return f"({bytes_to_unicode(self.left)} {bytes_to_unicode(self.right)})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class TokenRef:
    tokens: list[bytes]
    count: int = field(default=0)

    @property
    def bp_counts(self) -> dict[BytePair, int]:
        bp_to_counts: dict[BytePair, int] = {}

        for i in range(len(self.tokens) - 1):
            bp = BytePair(self.tokens[i], self.tokens[i + 1])
            if bp not in bp_to_counts:
                bp_to_counts[bp] = 0
            bp_to_counts[bp] += self.count

        return bp_to_counts

    def merge(self, bp: BytePair):
        new_tokens: list[bytes] = []

        idx = 0
        # Merge non-overlapping occurrences from left to right, preserving trailing token
        while idx < len(self.tokens):
            if idx < len(self.tokens) - 1:
                pair = BytePair(self.tokens[idx], self.tokens[idx + 1])
                if pair == bp:
                    new_tokens.append(bp.merged_bytes)
                    idx += 2
                    continue
            new_tokens.append(self.tokens[idx])
            idx += 1

        self.tokens = new_tokens


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_bytes(b: bytes) -> list[bytes]:
    return [bytes([bb]) for bb in list(b)]


def bytes_to_unicode(b: bytes) -> str:
    return "".join([BYTES_TO_UNICODE[bb] for bb in list(b)])
