# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is Stanford CS336 Spring 2025 Assignment 1: Basics - a deep learning course assignment focused on implementing core components of language models including BPE tokenization, transformer architectures, and training utilities.

## Environment Setup and Commands

### Package Management
This project uses `uv` for environment management. All commands should be prefixed with `uv run`:

```bash
# Install uv if not already installed
pip install uv  # or: brew install uv

# Run any Python file in the repo
uv run <python_file_path>
```

### Core Development Commands

```bash
# Run all unit tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tokenizer.py

# Run specific test
uv run pytest tests/test_tokenizer.py::test_encode

# Run tests with verbose output
uv run pytest -v

# Run tests and update snapshots (if using snapshot testing)
uv run pytest --update-snapshots

# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Create submission package
bash make_submission.sh
```

### Data Setup
```bash
# Download required datasets
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
```

## Architecture and Code Structure

### Main Implementation Areas

The assignment requires implementing core ML components in the following structure:

#### BPE Tokenization (`cs336_basics/bpe/`)
- **tokenizer.py**: BPE tokenizer implementation with encode/decode functionality
- **train.py**: BPE training algorithm for learning merges from corpus
- **pretokenize.py**: Pretokenization utilities for splitting text before BPE
- **utils.py**: Helper classes like BytePair and TokenRef for BPE operations

Key classes:
- `Tokenizer`: Main tokenizer class with encode/decode/encode_iterable methods
- Uses vocabulary (dict[int, bytes]) and merges (list[tuple[bytes, bytes]])
- Supports special tokens that are never split

#### Test Adapters (`tests/adapters.py`)
Central adapter file that connects your implementations to the test suite. Students must complete functions here to integrate their implementations:

- Neural network components: `run_linear`, `run_embedding`, `run_swiglu`, `run_rmsnorm`, `run_silu`
- Attention mechanisms: `run_scaled_dot_product_attention`, `run_multihead_self_attention`, `run_multihead_self_attention_with_rope`
- Position embeddings: `run_rope`
- Full models: `run_transformer_block`, `run_transformer_lm`
- Training utilities: `run_get_batch`, `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`
- Optimization: `get_adamw_cls`, `run_get_lr_cosine_schedule`
- Checkpointing: `run_save_checkpoint`, `run_load_checkpoint`
- Tokenization: `get_tokenizer`, `run_train_bpe`

### Key Implementation Details

#### Model Architecture
- Uses pre-norm transformer blocks with RMSNorm instead of LayerNorm
- SwiGLU activation function instead of standard FFN
- Rotary Position Embeddings (RoPE) with configurable theta parameter
- Multi-head self-attention with optimized batched implementation

#### Training Components
- AdamW optimizer implementation
- Cosine learning rate schedule with linear warmup
- Gradient clipping by L2 norm
- Memory-mapped dataset loading for efficient training

#### Testing Infrastructure
- Comprehensive unit tests in `tests/` directory
- Fixtures in `tests/fixtures/` with reference implementations
- Tests validate against GPT-2 tokenization for compatibility

## Important Implementation Notes

1. **Tensor Type Annotations**: The codebase uses jaxtyping for tensor shape annotations (e.g., `Float[Tensor, "batch seq_len d_model"]`)

2. **Memory Efficiency**: 
   - Use `encode_iterable` for large files that don't fit in memory
   - Dataset loading should use memory mapping (`np.memmap`)

3. **Numerical Stability**:
   - RMSNorm and SwiGLU require careful attention to numerical stability
   - Add epsilon values where specified in the assignment

4. **BPE Training**:
   - Lexicographic tiebreaking when counts are equal
   - Special tokens are added to vocabulary but treated as regular text during training
   - Pretokenization uses regex patterns to split text appropriately

5. **RoPE Implementation**:
   - Embedding dimension must match head dimension (d_model // num_heads)
   - Support for custom token positions
   - Pre-caching for efficiency

## Dependencies and Environment

- Python ≥3.11 required
- PyTorch ~2.6.0 (or ~2.2.2 for Intel Macs)
- Key libraries: einops, einx, jaxtyping, numpy, tiktoken, wandb
- Development tools: pytest, ruff, ipykernel

## Performance Considerations

- BPE training has time limits in tests - optimize for speed
- Large vocabulary training (50k+ tokens) should complete in reasonable time
- Use vectorized operations where possible
- Cache computed values (e.g., in RoPE) when beneficial