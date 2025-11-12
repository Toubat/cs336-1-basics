import modal

app = modal.App("dist-tokenizer")


image = modal.Image.debian_slim().uv_sync().add_local_python_source("cs336_basics")


source_volume = modal.Volume.from_name(
    "cs336",
    version=2,
    create_if_missing=True,
)

with image.imports():
    import glob
    import os
    import re
    import shutil
    import tempfile

    import numpy as np
    from tqdm import tqdm

    from cs336_basics.bpe.tokenizer import Tokenizer, encode_file_stream
    from cs336_basics.bpe.utils import find_chunk_boundaries


@app.function(
    image=image,
    volumes={"/source": source_volume},
)
def save_file_chunk(idx: int, start: int, end: int):
    """Save a file chunk to a separate file."""
    with open("/source/cs336-1-basics/data/owt_train.txt", "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    with open(f"/source/cs336-1-basics/data/owt_train_{idx}.txt", "wb") as f_out:
        f_out.write(chunk)

    source_volume.commit()

    print(f"Chunk {idx}: wrote {len(chunk):,} bytes (from byte {start:,} to {end:,})")
    return idx, len(chunk)


@app.function(
    image=image,
    volumes={"/source": source_volume},
)
def split_file(containers: int = 90):
    """Split file into chunks and save each chunk using distributed workers."""
    source_path = "/source/cs336-1-basics/data/owt_train.txt"
    split_token = b"<|endoftext|>"

    # Get file size
    file_size = os.stat(source_path).st_size
    print(f"File size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")
    print(f"Splitting into {containers} containers")

    # Calculate the boundaries of the file at special token locations
    with open(source_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, containers, split_token)

    print(f"Found {len(boundaries)} boundaries (creating {len(boundaries) - 1} chunks)")

    # Create list of (idx, start, end) tuples for each chunk
    file_boundaries: list[tuple[int, int, int]] = [
        (i, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
    ]

    # Call save_file_chunk in parallel for each boundary using Modal's distributed execution
    print(f"Spawning {len(file_boundaries)} parallel workers...")
    results = list(save_file_chunk.starmap(file_boundaries))

    # Print summary statistics
    total_bytes = sum(size for _, size in results)
    avg_chunk_size = total_bytes / len(results) if results else 0
    print(f"\nSuccessfully split file into {len(results)} chunks")
    print(f"Total: {total_bytes:,} bytes, {total_bytes / 1e9:.2f} GB")
    print(f"Average chunk size: {avg_chunk_size:,.0f} bytes, {avg_chunk_size / 1e6:.2f} MB")


@app.function(
    image=image,
    volumes={"/source": source_volume},
    cpu=64,
    timeout=3600,
)
def tokenize_file_chunk(idx: int, vocab_size: int = 10000):
    """Tokenize a single chunk file and save as .npy.

    Args:
        idx: Chunk index
        vocab_size: Vocabulary size for validation

    Returns:
        Tuple of (idx, num_tokens)
    """
    input_file = f"/source/cs336-1-basics/data/owt_train_{idx}.txt"
    output_file = f"/source/cs336-1-basics/data/owt_train_{idx}.npy"
    vocab_path = "/source/cs336-1-basics/tokenizers/owt_train_vocab.json"
    merges_path = "/source/cs336-1-basics/tokenizers/owt_train_merges.txt"
    special_tokens = ["<|endoftext|>"]

    if os.path.exists(output_file):
        print(f"Chunk {idx}: already tokenized, skipping")
        arr = np.load(output_file, mmap_mode="r")
        return idx, arr.shape[0]

    file_size = os.stat(input_file).st_size
    print(f"Chunk {idx}: tokenizing {file_size:,} bytes")

    # Collect tokens in memory (chunk is small enough)
    tokens = []

    for token in encode_file_stream(input_file, vocab_path, merges_path, special_tokens):
        if not isinstance(token, int | np.integer):
            raise ValueError(f"Token {token} (type: {type(token)}) is not an integer")

        if not (0 <= token < vocab_size):
            raise ValueError(f"Token {token} is out of range [0, {vocab_size})")

        tokens.append(token)

    # Convert to numpy array and save
    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_file, arr)

    source_volume.commit()

    print(f"Chunk {idx}: wrote {len(tokens):,} tokens to {output_file}")
    return idx, len(tokens)


@app.function(
    image=image,
    volumes={"/source": source_volume},
    timeout=3600,
)
def tokenize_all_chunks(vocab_size: int = 10000):
    """Tokenize all chunk files in parallel.

    Args:
        vocab_size: Vocabulary size for validation

    Returns:
        List of (idx, num_tokens) for each chunk
    """
    data_dir = "/source/cs336-1-basics/data"

    # Find all chunk files
    chunk_files = sorted(glob.glob(f"{data_dir}/owt_train_*.txt"))

    # Extract indices from filenames
    indices = []
    for f in chunk_files:
        match = re.search(r"owt_train_(\d+)\.txt$", f)
        if match:
            indices.append(int(match.group(1)))
        else:
            raise ValueError(f"No match found for file {f}")

    if not indices:
        raise ValueError(f"No chunk files found matching pattern owt_train_*.txt in {data_dir}")

    print(f"Found {len(indices)} chunk files to tokenize")
    print(f"Chunk indices: {min(indices)} to {max(indices)}")

    # Tokenize all chunks in parallel using Modal's distributed execution
    print(f"Spawning {len(indices)} parallel workers to tokenize chunks...")
    results = list(tokenize_file_chunk.starmap([(idx, vocab_size) for idx in indices]))

    # Print summary statistics
    total_tokens = sum(num_tokens for _, num_tokens in results)
    avg_tokens = total_tokens / len(results) if results else 0

    print(f"\nSuccessfully tokenized {len(results)} chunks")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per chunk: {avg_tokens:,.0f}")

    return results


@app.function(
    image=image,
    volumes={"/source": source_volume},
    timeout=36000,  # 1 hour timeout for large merges
    cpu=64,
)
def merge_tokenized_files():
    """Merge all tokenized chunk .npy files into a single file.

    Returns:
        Total number of tokens in the merged file
    """
    data_dir = "/source/cs336-1-basics/data"
    output_file = f"{data_dir}/owt_train.npy"

    # Find all chunk .npy files
    chunk_files = sorted(glob.glob(f"{data_dir}/owt_train_*.npy"))

    # Extract indices and sort by index
    file_info = []
    for f in chunk_files:
        match = re.search(r"owt_train_(\d+)\.npy$", f)
        if match:
            idx = int(match.group(1))
            file_info.append((idx, f))
        else:
            raise ValueError(f"No match found for file {f}")

    # Sort by index to ensure correct order
    file_info.sort(key=lambda x: x[0])

    if not file_info:
        raise ValueError(f"No chunk .npy files found in {data_dir}")

    print(f"Found {len(file_info)} chunk .npy files to merge")
    print(f"Chunk indices: {file_info[0][0]} to {file_info[-1][0]}")

    # Read each npy file to get size
    chunk_sizes = []
    total_tokens = 0

    print("Reading chunk sizes...")
    for idx, filepath in tqdm(file_info, desc="Reading sizes"):
        arr = np.load(filepath, mmap_mode="r")
        size = arr.shape[0]
        chunk_sizes.append(size)
        total_tokens += size
        print(f"Chunk {idx}: {size:,} tokens")

    print(f"Total tokens to merge: {total_tokens:,}")

    # Allocate memmap with total size in a temp file
    with tempfile.NamedTemporaryFile(suffix=".npy", dir="/tmp", delete=False) as temp_f:
        temp_file = temp_f.name

    print(f"Allocating merged array with {total_tokens:,} tokens...")
    merged_arr = np.lib.format.open_memmap(temp_file, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    # Copy each chunk into the merged array
    pos = 0
    print("Merging chunks...")
    for (idx, filepath), size in tqdm(zip(file_info, chunk_sizes), total=len(file_info), desc="Merging"):
        chunk_arr = np.load(filepath)
        merged_arr[pos : pos + size] = chunk_arr
        pos += size
        merged_arr.flush()
        print(f"Merged chunk {idx}: {size:,} tokens (total so far: {pos:,})")

    merged_arr.flush()

    # Move temp file to final location
    print(f"Moving merged file to {output_file}")
    shutil.move(temp_file, output_file)

    source_volume.commit()

    print(f"Successfully merged {len(file_info)} chunks into {output_file}")
    print(f"Total tokens: {total_tokens:,}")

    return total_tokens


@app.function(
    image=image,
    volumes={"/source": source_volume},
    timeout=36000,  # 1 hour timeout for large merges
    cpu=64,
)
def cleanup():
    # remove all chunk text / npy files
    data_dir = "/source/cs336-1-basics/data"
    tokenizer_dir = "/source/cs336-1-basics/tokenizers"

    chunk_files = sorted(glob.glob(f"{data_dir}/owt_train_*.txt"))
    chunk_files.extend(glob.glob(f"{data_dir}/owt_train_*.npy"))
    for file in chunk_files:
        os.remove(file)

    source_volume.commit()

    tokenizer = Tokenizer.from_file(
        f"{tokenizer_dir}/owt_train_vocab.json",
        f"{tokenizer_dir}/owt_train_merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    # print first and last 100 tokens
    data = np.lib.format.open_memmap(f"{data_dir}/owt_train.npy", mode="r", dtype=np.uint16)
    print("First 100 tokens:")
    print(tokenizer.decode(data[:100].tolist()))
    print("Last 100 tokens:")
    print(tokenizer.decode(data[-100:].tolist()))

    print("Cleaned up all chunk text / npy files")


@app.local_entrypoint()
def main(
    containers: int = 90,
    vocab_size: int = 10000,
):
    """Run the distributed tokenization pipeline.

    Args:
        containers: Number of chunks to split the file into (for 'split' stage)
        vocab_size: Vocabulary size for validation (for 'tokenize' stage)
    """

    print("Running full pipeline: split -> tokenize -> merge")
    print("\n" + "=" * 60)
    print("Stage 1: Splitting file into chunks...")
    print("=" * 60)
    split_file.remote(containers)

    print("\n" + "=" * 60)
    print("Stage 2: Tokenizing all chunks in parallel...")
    print("=" * 60)
    tokenize_all_chunks.remote(vocab_size)

    print("\n" + "=" * 60)
    print("Stage 3: Merging tokenized chunks...")
    print("=" * 60)
    merge_tokenized_files.remote()

    print("\n" + "=" * 60)
    print("Cleaning up...")
    print("=" * 60)
    cleanup.remote()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
