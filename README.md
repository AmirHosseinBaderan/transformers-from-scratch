# Project Layers — transformers-from-scratch

This document describes every layer of the `transformers-from-scratch` project, from data ingestion to model training and inference. Each layer is organized by responsibility and explains **what** it does, **why** it exists, and **how** it connects to the other layers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Point (main_gpt.py)                    │
│  Runs data pipeline → then trains the model                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │  Data Layer   │  │  Model Layer    │  │  Training Layer  │
   │  (common/data)│  │  (common/nn)    │  │  (common/training)│
   └──────────────┘  └─────────────────┘  └──────────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │  Config Layer │  │  Utility Layer  │  │  MiniGPT App     │
   │  (common/configs)│ (common/utils) │  │  (mini_gpt/)     │
   └──────────────┘  └─────────────────┘  └──────────────────┘
```

---

## 1. Data Layer (`common/data/`)

The data layer is responsible for reading raw text, building a vocabulary, tokenizing text, and producing binary files that the model can consume efficiently.

### 1.1 Readers (`common/data/readers/`)

#### `reader.py` — `Reader` (ABC)
- **What**: Abstract base class for all data readers.
- **Why**: Defines a contract (`read() -> list[str]`) so that any reader implementation can be swapped without changing downstream code.
- **How it works**: Subclasses must implement `read()` which returns a list of text strings.

#### `csv_reader.py` — `CSVReader`
- **What**: Streams rows from a CSV file one at a time.
- **Why**: The raw dataset (`common/data/raw/train.csv`, `validation.csv`) is stored as CSV. Loading it all into memory at once would be wasteful for large files.
- **How it works**: Uses Python's `csv.DictReader` to yield each row's `text` column as a string. It skips empty rows and missing columns.

### 1.2 Vocabulary (`common/data/vocabulary.py`) — `Vocabulary`
- **What**: A bidirectional mapping between tokens (strings) and integer IDs.
- **Why**: Neural networks operate on integers, not strings. The vocabulary converts characters (or tokens) to IDs and back.
- **How it works**:
  - `add_token(token)` — assigns the next available integer ID to a new token.
  - `token_to_id(token)` — looks up a token's ID; falls back to `<UNK>` if unknown.
  - `id_to_token(id)` — looks up the string for a given ID.
  - `encode(tokens)` / `decode(token_ids)` — batch conversion helpers.
  - `save(path)` / `load(path)` — persists the vocabulary as JSON so it can be reused without rebuilding.
  - `size` — returns the total number of tokens.

### 1.3 Tokenizers (`common/data/`)

#### `tokenizer.py` — `Tokenizer` (ABC)
- **What**: Abstract base class for all tokenizers.
- **Why**: Provides a consistent interface (`encode`, `decode`, `vocabulary`) so different tokenization strategies can be used interchangeably.

#### `character_tokenizer.py` — `CharacterTokenizer`
- **What**: A concrete tokenizer that operates at the character level.
- **Why**: Character-level tokenization is simple, has no unknown tokens (every character is in the vocab), and works well for small datasets.
- **How it works**:
  - `encode(text)` — converts each character to its vocabulary ID.
  - `encode_iterable(texts)` — a generator that yields token IDs for multiple texts, optionally appending `<EOS>` after each.
  - `decode(token_ids)` — joins the characters back into a string.

### 1.4 Vocabulary Builders (`common/data/builders/`)

#### `vocabulary_builder.py` — `VocabularyBuilder` (ABC)
- **What**: Abstract base for vocabulary construction strategies.
- **Why**: Allows different building strategies (character-level, BPE, word-level) to be plugged in.

#### `character_vocabulary_builder.py` — `CharacterVocabularyBuilder`
- **What**: Builds a vocabulary by counting character frequencies in the training text.
- **Why**: The simplest approach — scan all text, count characters, sort by frequency, assign IDs.
- **How it works**:
  1. Uses `collections.Counter` to count every character across all texts.
  2. Creates a `Vocabulary` with special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
  3. Adds characters in descending frequency order.

### 1.5 Dataset Preparation (`common/data/preprocessing/`)

#### `dataset_preparer.py` — `DatasetPreparer`
- **What**: Orchestrates the two-step process of building a vocabulary and encoding text into binary files.
- **Why**: Separates the one-time data preparation from the training loop, so data is pre-processed into an efficient format (`np.uint16` binary) that `TextDataset` can memory-map.
- **How it works**:
  - `prepare_vocabulary(texts, output_path)` — builds vocabulary from text stream and saves to `vocab.json`.
  - `encode_to_file(texts, tokenizer, output_path)` — tokenizes text and writes IDs as a binary array (`array('H')` = unsigned short = `np.uint16`) to `.bin` files. Uses a buffer of 65536 elements for efficient I/O.

### 1.6 Dataset (`common/data/dataset.py`) — `TextDataset`
- **What**: A PyTorch `Dataset` that memory-maps the binary token files with configurable stride.
- **Why**: Memory-mapping (`np.memmap`) avoids loading the entire dataset into RAM. Only the small slices needed for each batch are read. The stride parameter allows overlapping windows for more granular training samples.
- **How it works**:
  - `__init__(path, block_size, stride=None)` — opens the `.bin` file as a read-only `np.memmap` of `uint16`. If `stride` is not provided, it defaults to `block_size` (non-overlapping windows).
  - `__len__()` — returns `(len(tokens) - block_size) // stride`, the number of possible strided input-target pairs.
  - `__getitem__(index)` — computes `start = index * stride`, then returns `(x, y)` where `x = tokens[start:start+block_size]` and `y = tokens[start+1:start+block_size+1]`. This is the standard autoregressive setup: predict the next token given the previous `block_size` tokens.

---

## 2. Model Layer (`common/nn/`)

The model layer implements the neural network components of a GPT-style decoder-only transformer.

### 2.1 Embedding Layers (`common/nn/layers/`)

#### `embedding.py` — `TokenEmbedding`
- **What**: A simple `nn.Embedding` wrapper that converts token IDs to dense vectors.
- **Why**: The first step in the model — turn discrete token indices into continuous vectors the network can process.
- **How it works**: `nn.Embedding(vocab_size, embedding_dim)` — a lookup table of shape `(vocab_size, embedding_dim)`.

#### `positional_embedding.py` — `PositionalEmbedding`
- **What**: Learnable positional embeddings that encode the position of each token in the sequence.
- **Why**: Transformers have no inherent notion of order (unlike RNNs). Positional embeddings give the model information about where each token sits in the sequence.
- **How it works**: Creates an `nn.Embedding(block_size, embedding_dim)` lookup. In `forward(x)`, it generates position indices `0..seq_length-1` and looks up their embeddings. The input `x` is only used to determine the sequence length.

#### `input_embedding.py` — `InputEmbedding`
- **What**: Combines token embeddings and positional embeddings via addition.
- **Why**: The model needs both *what* each token is (token embedding) and *where* it is (positional embedding) to understand the sequence.
- **How it works**: `forward(x)` returns `token_embedding(x) + position_embedding(x)`. Both have the same shape `(batch, seq_len, embedding_dim)`.

### 2.2 Attention Layers (`common/nn/layers/`)

#### `attention.py` — `SelfAttention`
- **What**: A single-head self-attention mechanism with causal masking.
- **Why**: This is the core building block of the transformer — it allows each token to attend to all previous tokens (causal = no peeking at the future).
- **How it works**:
  1. Projects input `x` into Query, Key, Value vectors via three `nn.Linear(head_dim, head_dim)` layers.
  2. Computes attention scores: `Q @ K^T / sqrt(head_dim)`.
  3. Applies a causal upper-triangular mask (sets future positions to `-inf`).
  4. Applies `softmax` to get attention weights.
  5. Applies dropout to weights.
  6. Returns `weights @ V`.

#### `multi_head_attention.py` — `MultiHeadAttention`
- **What**: Multi-head attention that runs several `SelfAttention` heads in parallel and concatenates their outputs.
- **Why**: Multiple attention heads allow the model to attend to different types of relationships (syntax, semantics, position) simultaneously.
- **How it works**:
  1. Projects input into Q, K, V all at once via a single `nn.Linear(embedding_dim, embedding_dim * 3)` (more efficient than three separate linear layers).
  2. Splits Q, K, V into `num_heads` chunks of dimension `head_dim = embedding_dim // num_heads`.
  3. Reshapes and transposes to `(batch, num_heads, seq_len, head_dim)`.
  4. Computes attention scores with causal masking (same as `SelfAttention`).
  5. Applies softmax and dropout.
  6. Multiplies attention weights by V.
  7. Transposes back and concatenates heads.
  8. Projects the concatenated output via `nn.Linear(embedding_dim, embedding_dim)`.

### 2.3 Feed-Forward Layer (`common/nn/layers/feed_forward.py`) — `FeedForward`
- **What**: A position-wise feed-forward network (FFN) with two linear layers and GELU activation.
- **Why**: After attention, each position needs a non-linear transformation to mix information. The FFN is applied identically to each position (hence "position-wise").
- **How it works**: `nn.Sequential(Linear(embedding_dim, embedding_dim*4), GELU(), Linear(embedding_dim*4, embedding_dim), Dropout(dropout))`. The expansion ratio of 4x is standard in GPT architectures.

### 2.4 Normalization (`common/nn/layers/layer_norm.py`) — `LayerNorm`
- **What**: A thin wrapper around `nn.LayerNorm`.
- **Why**: Layer normalization stabilizes training by normalizing activations across the feature dimension. It's applied before attention and feed-forward in the decoder block (pre-norm architecture).
- **How it works**: Delegates directly to `nn.LayerNorm(embedding_dim, eps=1e-5)`.

### 2.5 Blocks (`common/nn/blocks/`)

#### `decoder_block.py` — `DecoderBlock`
- **What**: A single GPT decoder block — the fundamental repeating unit of the transformer.
- **Why**: Stacking multiple decoder blocks creates the deep transformer that can learn complex patterns.
- **How it works** (pre-norm architecture with residual connections):
  1. **Attention sub-layer**: `x = x + Attention(LayerNorm(x))` — normalize, attend, add residual.
  2. **Feed-forward sub-layer**: `x = x + FeedForward(LayerNorm(x))` — normalize, transform, add residual.
  3. Supports **gradient checkpointing** (trades compute for memory by recomputing activations during backward pass instead of storing them).

### 2.6 Blocks (`common/nn/blocks/`)

#### `encoder_block.py` — (Empty placeholder)
- **What**: Reserved for future encoder block implementation.
- **Why**: The project currently only implements a decoder-only (GPT) architecture, but the placeholder allows easy extension to encoder-decoder models later.

---

## 3. Training Layer (`common/training/`)

The training layer provides reusable infrastructure for training any model in this project.

### 3.1 Loss (`common/training/losses.py`) — `LanguageModelLoss`
- **What**: Cross-entropy loss for autoregressive language modeling.
- **Why**: The standard loss for next-token prediction. It measures how well the model's predicted probability distribution matches the actual next token.
- **How it works**: Flattens `(batch, seq, vocab_size)` logits to `(batch*seq, vocab_size)` and `(batch, seq)` targets to `(batch*seq,)`, then computes `nn.CrossEntropyLoss()`.

### 3.2 Checkpointing (`common/training/checkpoint.py`) — `CheckpointManager`
- **What**: Saves and loads model checkpoints during training.
- **Why**: Enables resuming training after interruption and keeps the best model based on validation loss.
- **How it works**:
  - `save(epoch, train_loss, val_loss, is_best)` — saves a dict with model state, optimizer state, loss metrics, and early stopping state to `checkpoint_epoch_N.pt`. If `is_best`, also saves to `best_model.pt`.
  - `load(path)` — restores model and optimizer state from a checkpoint file.
  - `load_latest()` — finds the most recent checkpoint by epoch number and loads it.
  - `_cleanup_old_checkpoints()` — keeps only the last `keep_last_n` checkpoints to avoid disk bloat.

### 3.3 Early Stopping (`common/training/early_stopping.py`) — `EarlyStopping`
- **What**: Monitors a validation metric and stops training when it stops improving.
- **Why**: Prevents overfitting and saves compute when the model has converged.
- **How it works**:
  - On each call `__call__(val_loss)`, compares the current score to the best score.
  - If improvement ≥ `min_delta`, resets the counter; otherwise increments it.
  - When `counter >= patience`, sets `early_stop = True`.
  - Supports both `min` mode (lower is better, e.g., loss) and `max` mode (higher is better, e.g., accuracy).
  - `state_dict()` / `load_state_dict()` — serializes/deserializes state for checkpointing.

### 3.4 TensorBoard Logging (`common/training/tensorboard_logger.py`) — `TensorBoardLogger`
- **What**: Wraps `torch.utils.tensorboard.SummaryWriter` to log training metrics.
- **Why**: Provides visual monitoring of loss curves, learning rate, and other metrics during training.
- **How it works**: `log_training_loss()`, `log_validation_loss()`, and `log_learning_rate()` delegate to `log_scalar()` which calls `writer.add_scalar()`.

### 3.5 Trainer (`common/training/trainer.py`) — `Trainer` (placeholder)
- **What**: Reserved for a generic trainer class. Currently the training loop is implemented in `mini_gpt/trainer.py` instead.
- **Why**: The placeholder exists for future generalization of the training logic.

---

## 4. Config Layer (`common/configs/`)

### `model_config.py` — `ModelConfig`
- **What**: Central configuration class with auto-detected optimal settings based on available hardware.
- **Why**: Different machines have different GPU/CPU memory; hardcoding batch size and other settings would cause OOM errors on some machines and underutilization on others.
- **How it works**:
  - At import time, `_get_optimal_config()` detects CUDA availability and GPU/CPU memory.
  - Based on memory, it auto-selects `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `USE_MIXED_PRECISION`, `USE_GRADIENT_CHECKPOINTING`, etc.
  - **Key change**: `BATCH_SIZE` is now uniformly `32` across all memory tiers (previously varied from 2–16). This reflects a design decision to use a fixed batch size with gradient accumulation steps to control effective batch size.
  - Model architecture defaults: `EMBEDDING_DIM=128`, `BLOCK_SIZE=128`, `NUM_HEADS=4`, `NUM_LAYERS=4`, `DROPOUT=0.1`.
  - Training defaults: `LEARNING_RATE=3e-4`, `EPOCHS=10`.
  - **Training step settings**: `STEPS_PER_EPOCH=5000`, `VAL_STEPS=200` — fixed step counts that decouple epoch size from dataset length, providing deterministic and reproducible training loops.
  - Memory optimization settings (auto-configured): `USE_MIXED_PRECISION`, `USE_GRADIENT_CHECKPOINTING`, `GRADIENT_CLIP_NORM`.
  - DataLoader settings (auto-configured): `PIN_MEMORY`, `NUM_WORKERS`, `PREFETCH_FACTOR`.
  - Checkpoint settings: `CHECKPOINT_DIR`, `KEEP_LAST_N_CHECKPOINTS`.
  - Early stopping settings: `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`.
  - TensorBoard settings: `TENSORBOARD_LOG_DIR`.

### `data_config.py` — `DataConfig`
- **What**: Legacy data configuration (batch size, block size, shuffle, num workers, pin memory).
- **Why**: Was used before `ModelConfig` absorbed all auto-detection logic. Kept for backward compatibility.

---

## 5. Utility Layer (`common/utils/`)

### `logger.py` — `logger`
- **What**: A module-level Python `logging.Logger` configured with `INFO` level and a timestamp format.
- **Why**: Provides consistent, timestamped log output across all modules without each module needing to configure its own logger.

### `device.py` — (Empty placeholder)
- **What**: Reserved for future device management utilities.

### `seed.py` — (Empty placeholder)
- **What**: Reserved for future reproducibility utilities (setting random seeds).

---

## 6. Application Layer (`mini_gpt/`)

The `mini_gpt/` directory contains the concrete application that uses all the shared layers above.

### `config.py` — `GPTConfig`
- **What**: Re-exports `ModelConfig` as `GPTConfig` for backward compatibility.
- **Why**: Allows the mini_gpt package to reference configuration without importing from the shared `common` package directly.

### `model.py` — `MiniGPT`
- **What**: The complete GPT model — embedding + N decoder blocks + layer norm + language model head.
- **Why**: This is the core model that the project trains and uses for text generation.
- **How it works**:
  1. `InputEmbedding` converts token IDs to vectors.
  2. `N` `DecoderBlock` layers process the sequence.
  3. `LayerNorm` normalizes the output.
  4. `lm_head` (a linear layer with no bias) projects back to vocabulary size to produce logits.
  5. `_init_weights()` initializes linear layers with `N(0, 0.02)` and embeddings with `N(0, 0.02)`.
  6. `forward(input_ids, targets)` — runs the forward pass; if `targets` is provided, computes cross-entropy loss.
  7. `generate(input_ids, max_new_tokens, temperature, do_sample)` — autoregressive generation loop: for each new token, runs the model on the current context, takes the last logits, applies temperature scaling, and either samples (with `do_sample=True`) or takes the argmax.

### `train.py` — Training Entry Point
- **What**: The script that sets up all components and starts training.
- **Why**: Orchestrates dataset loading, model creation, optimizer setup, and the training loop.
- **How it works**:
  1. Loads `TextDataset` from binary files.
  2. Creates `DataLoader` instances with `RandomSampler` using fixed step counts (`STEPS_PER_EPOCH`, `VAL_STEPS`) for deterministic epoch sizes.
  3. Uses `pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available()` for dynamic pin memory based on device availability.
  4. Loads vocabulary and creates `MiniGPT` model.
  5. Sets up `AdamW` optimizer, `LanguageModelLoss` criterion, `CheckpointManager`, `EarlyStopping`, and `TensorBoardLogger`.
  6. Creates a `Trainer` and calls `trainer.train()`.
  7. Handles OOM errors with actionable suggestions.
  8. Performs aggressive memory cleanup in `finally` block.

### `trainer.py` — `Trainer`
- **What**: The training loop orchestrator.
- **Why**: Separates the training loop logic from the setup logic in `train.py`.
- **How it works**:
  1. For each epoch: runs `train_one_epoch()`, then `validate_one_epoch()`.
  2. Performs aggressive memory cleanup between phases.
  3. Logs metrics to TensorBoard.
  4. Saves checkpoints and checks early stopping.
  5. Cleans up at the end.

### `train_one_epoch.py` — `train_one_epoch()`
- **What**: Runs one epoch of training with gradient accumulation, mixed precision, and gradient clipping.
- **Why**: These are standard techniques for training large models on limited hardware.
- **How it works**:
  1. Sets model to train mode.
  2. Iterates over batches, moving data to device.
  3. If mixed precision: uses `autocast()` and `GradScaler` for loss scaling.
  4. Accumulates gradients over `gradient_accumulation_steps` before calling `optimizer.step()`.
  5. Clips gradients if `gradient_clip_norm` is set.
  6. Deletes tensors after each batch and periodically clears CUDA cache.
  7. Handles remaining gradients if batch count is not divisible by accumulation steps.

### `validate_one_epoch.py` — `validate_one_epoch()`
- **What**: Runs one epoch of validation.
- **Why**: Evaluates model performance on held-out data without updating weights.
- **How it works**: Similar to training but with `torch.no_grad()`, no optimizer steps, and no gradient accumulation.

### `generate.py` — Text Generation Script
- **What**: Loads a trained checkpoint and generates text from a prompt.
- **Why**: Provides a simple way to use the trained model for inference.
- **How it works**:
  1. Loads vocabulary and tokenizer.
  2. Loads the best model checkpoint.
  3. Encodes a prompt string to token IDs.
  4. Runs `model.generate()` with temperature=0.8 and sampling.
  5. Decodes the generated token IDs back to text and prints.

---

## 7. Entry Point (`main_gpt.py`)

- **What**: The top-level pipeline script that runs the data pipeline and then training sequentially.
- **Why**: Provides a single command to go from raw CSV data to a trained model.
- **How it works**: Calls `data_pipeline.py` as a subprocess, then calls `python -m mini_gpt.train` as a subprocess.

---

## 8. Data Pipeline (`data_pipeline.py`)

- **What**: The standalone data preparation script.
- **Why**: Separates data preparation from training so they can be run independently.
- **How it works**:
  1. Creates a `DatasetPreparer` with a `CharacterVocabularyBuilder`.
  2. Reads the training CSV and builds the vocabulary, saving it to `vocab.json`.
  3. Creates a `CharacterTokenizer` from the vocabulary.
  4. Encodes the training and validation CSVs into binary `.bin` files.

---

## Summary of Data Flow

```
CSV files (raw/)
    │
    ▼
CSVReader → CharacterVocabularyBuilder → Vocabulary (vocab.json)
    │
    ▼
CharacterTokenizer → DatasetPreparer → Binary files (train.bin, validation.bin)
    │
    ▼
TextDataset (memory-mapped, strided) → DataLoader → (x, y) batches
    │
    ▼
MiniGPT model → LanguageModelLoss → optimizer.step()
    │
    ▼
CheckpointManager saves best model → TensorBoardLogger tracks metrics
    │
    ▼
generate.py loads best model → text generation
```
