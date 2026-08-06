# MiniT5 — From-Scratch Encoder-Decoder Transformer

MiniT5 is a lightweight, educational implementation of the T5 (Text-to-Text Transfer Transformer) architecture built entirely from scratch using PyTorch. It is designed for English-to-Farsi (Persian) translation and serves as a hands-on learning resource for understanding encoder-decoder transformer models.

---

## Table of Contents

1. [What is T5?](#what-is-t5)
2. [How T5 Works](#how-t5-works)
3. [Where T5 is Used](#where-t5-is-used)
4. [Project Overview](#project-overview)
5. [Architecture](#architecture)
6. [Standard vs. This Implementation](#standard-vs-this-implementation)
7. [File-by-File Description](#file-by-file-description)
8. [Training & Inference](#training--inference)
9. [Configuration](#configuration)

---

## What is T5?

**T5** stands for **Text-to-Text Transfer Transformer**. It was introduced by Google Research in 2020 in the paper *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"* (Raffel et al.).

The core idea behind T5 is simple but powerful: **every NLP task is framed as a text-to-text problem**. Whether the task is translation, summarization, question answering, or classification, the model always takes text as input and produces text as output. This unified framework allows a single model architecture and training procedure to be applied across a wide variety of tasks.

### Key Characteristics of T5

- **Encoder-Decoder Architecture**: T5 uses a transformer encoder to process the input sequence and a transformer decoder to generate the output sequence.
- **Text-to-Text Framework**: Inputs are prefixed with a task descriptor (e.g., `translate English to French: ...`) and outputs are plain text.
- **Large-Scale Pre-training**: T5 is pre-trained on a massive corpus (the Colossal Clean Crawled Corpus — C4) using a span-corruption objective, where random spans of text are masked and the model learns to reconstruct them.
- **Scalable Variants**: T5 comes in multiple sizes — Small, Base, Large, XL, XXL — allowing practitioners to choose the right trade-off between performance and computational cost.

---

## How T5 Works

### High-Level Data Flow

```
Input Text (with task prefix)
        │
        ▼
   ┌─────────┐
   │ Encoder │  ← Token Embedding + Positional Encoding + N Encoder Blocks
   └────┬────┘
        │  Encoder Output (context vector)
        ▼
   ┌─────────┐
   │ Decoder │  ← Token Embedding + Positional Encoding + N Decoder Blocks
   └────┬────┘
        │
        ▼
   Output Text (generated token by token)
```

### The Encoder

The encoder reads and understands the input sequence:

1. **Token Embedding**: Each input token is converted into a dense vector.
2. **Positional Encoding**: Since transformers have no inherent sense of order, positional information is added to the embeddings.
3. **Encoder Blocks**: A stack of identical layers, each containing:
   - **Multi-Head Self-Attention**: Allows each token to attend to all other tokens in the input sequence, capturing relationships and context.
   - **Feed-Forward Network**: A position-wise fully connected network that transforms each token's representation independently.
   - **Layer Normalization & Residual Connections**: Stabilize training and enable deep networks.

The output of the encoder is a sequence of context vectors — one for each input token — that encode the meaning of the entire input.

### The Decoder

The decoder generates the output sequence one token at a time:

1. **Token Embedding & Positional Encoding**: Similar to the encoder, but for the output sequence.
2. **Masked Self-Attention**: Each token can only attend to previous tokens (causal masking), ensuring the model generates text autoregressively.
3. **Cross-Attention**: The decoder attends to the encoder's output, allowing it to focus on relevant parts of the input when generating each output token.
4. **Feed-Forward Network**: Further transforms the representations.
5. **Output Projection**: The final hidden states are projected to vocabulary size, producing logits over the next possible token.

### Training Objective

T5 is trained using **teacher forcing**: the decoder receives the correct previous tokens during training, and the model learns to predict the next token. The loss is typically cross-entropy between the predicted logits and the target tokens.

---

## Where T5 is Used

T5's text-to-text framework makes it incredibly versatile. Some common applications include:

| Task | Example Input | Example Output |
|------|---------------|----------------|
| **Translation** | `translate English to French: Hello world` | `Bonjour le monde` |
| **Summarization** | `summarize: [long article]` | `Short summary...` |
| **Question Answering** | `question: What is the capital of France? context: ...` | `Paris` |
| **Text Classification** | `cola sentence: This is a test.` | `1` (grammatical) |
| **Natural Language Inference** | `mnli premise: ... hypothesis: ...` | `entailment / neutral / contradiction` |
| **Coreference Resolution** | `resolve: ...` | `Resolved text...` |
| **Sentence Similarity** | `stsb sentence1: ... sentence2: ...` | `Similarity score` |

In this project, MiniT5 is specifically trained for **English-to-Farsi translation**.

---

## Project Overview

MiniT5 is a simplified, educational implementation of T5. It is not intended to match the performance of the full T5 model, but rather to demonstrate the core concepts of the encoder-decoder transformer architecture in a clear, readable codebase.

### Design Goals

- **From-scratch implementation**: No reliance on Hugging Face Transformers or other high-level libraries for the model itself.
- **Educational clarity**: Code is structured to be easy to read and understand, with each component isolated in its own module.
- **Memory-efficient training**: Includes gradient checkpointing, mixed precision, and gradient accumulation to run on limited hardware.
- **Practical application**: Trained on a real English-to-Farsi translation dataset.

---

## Architecture

### Model Architecture Diagram

```
                        ┌──────────────────────────────────────────────┐
                        │              MiniT5 (nn.Module)              │
                        └──────────────────┬───────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
            │   Encoder    │      │   Decoder    │      │ Output Projection │
            │  (T5Encoder) │      │  (T5Decoder) │      │   (Linear)        │
            └──────┬───────┘      └──────┬───────┘      └──────────────────┘
                   │                     │
         ┌─────────┴─────────┐         │
         ▼                   ▼         ▼
   ┌──────────┐      ┌──────────┐  ┌──────────┐
   │ Embedding│      │ Position │  │ Embedding│
   │ (Token)  │      │ Encoding │  │ (Token)  │
   └────┬─────┘      └────┬─────┘  └────┬─────┘
        │                 │             │
        ▼                 ▼             ▼
   ┌─────────────────────────────────────────────┐
   │          Encoder Blocks (x N)               │
   │  ┌───────────────────────────────────────┐  │
   │  │  Multi-Head Self-Attention            │  │
   │  │  + LayerNorm + Residual               │  │
   │  ├───────────────────────────────────────┤  │
   │  │  Feed-Forward (Linear → GELU → Linear)│  │
   │  │  + LayerNorm + Residual               │  │
   │  └───────────────────────────────────────┘  │
   └─────────────────────────────────────────────┘
                   │
                   │  Encoder Output
                   ▼
   ┌─────────────────────────────────────────────┐
   │          Decoder Blocks (x N)               │
   │  ┌───────────────────────────────────────┐  │
   │  │  Masked Multi-Head Self-Attention     │  │
   │  │  + LayerNorm + Residual               │  │
   │  ├───────────────────────────────────────┤  │
   │  │  Cross-Attention (to Encoder Output)  │  │
   │  │  + LayerNorm + Residual               │  │
   │  ├───────────────────────────────────────┤  │
   │  │  Feed-Forward (Linear → GELU → Linear)│  │
   │  │  + LayerNorm + Residual               │  │
   │  └───────────────────────────────────────┘  │
   └─────────────────────────────────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │  LayerNorm   │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │ Linear       │
            │ (embed → vocab)│
            └──────┬───────┘
                   │
                   ▼
               Logits
```

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `vocab_size` | ~100 (character-level) | Size of the token vocabulary |
| `embedding_dim` | 128 | Dimension of token embeddings |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 4 | Number of encoder/decoder layers |
| `ff_hidden_dim` | 512 | Hidden dimension of feed-forward networks (4× embedding_dim) |
| `max_length` | 64 | Maximum sequence length |
| `dropout` | 0.1 | Dropout probability |

---

## Standard vs. This Implementation

MiniT5 follows the **core T5 architecture** but makes several simplifications and deviations for educational and practical reasons.

### What is Standard (Follows T5)

| Component | Status | Notes |
|-----------|--------|-------|
| Encoder-Decoder Structure | ✅ Standard | Matches T5's encoder-decoder design |
| Multi-Head Self-Attention | ✅ Standard | Same scaled dot-product attention |
| Masked Self-Attention in Decoder | ✅ Standard | Causal masking for autoregressive generation |
| Cross-Attention in Decoder | ✅ Standard | Decoder attends to encoder output |
| Pre-Norm Architecture | ✅ Standard | LayerNorm before attention/FFN sub-layers |
| Residual Connections | ✅ Standard | Around each sub-layer |
| GELU Activation | ✅ Standard | Used in feed-forward networks |
| Feed-Forward Network | ✅ Standard | Linear → GELU → Linear with 4× expansion |
| Sinusoidal Positional Encoding | ✅ Standard | Like the original Transformer (T5 uses relative position bias instead) |
| Output Projection to Vocabulary | ✅ Standard | Linear layer projecting to vocab size |

### Deviations from Standard T5

| Component | Status | Explanation |
|-----------|--------|-------------|
| **Positional Encoding** | ⚠️ Different | Uses sinusoidal positional encoding (like original Transformer). Standard T5 uses **relative positional bias** instead of absolute positional embeddings. |
| **Tokenization** | ⚠️ Different | Uses **character-level tokenization**. Standard T5 uses **SentencePiece BPE** (Byte Pair Encoding) with a vocabulary of ~32,000 tokens. |
| **Embedding Sharing** | ⚠️ Different | Encoder and decoder have **separate** token embeddings. Standard T5 **shares** the token embedding matrix between encoder and decoder, and also ties the output projection to the input embedding. |
| **Normalization** | ⚠️ Different | Uses standard **LayerNorm**. Standard T5 uses **RMSNorm** (Root Mean Square Layer Normalization), which is simpler and more stable. |
| **Linear Layer Biases** | ⚠️ Different | Linear layers include **bias terms** by default. Standard T5 **removes biases** from all linear layers (Q, K, V, output projections, FFN). |
| **Model Scale** | ⚠️ Different | "Mini" version with 4 layers, 4 heads, 128-dim. Standard T5 variants are much larger (e.g., T5-Base: 12 layers, 12 heads, 768-dim). |
| **Pre-training Objective** | N/A | MiniT5 is trained from scratch on translation data. Standard T5 is pre-trained on a massive corpus with span corruption. |

### Summary

MiniT5 captures the **essential architectural patterns** of T5 — the encoder-decoder structure, multi-head attention, cross-attention, and pre-norm design — while simplifying tokenization, normalization, and model scale for educational clarity and practical training on limited hardware.

---

## File-by-File Description

### Root-Level Files

#### [`main_train.py`](main_train.py) — Training Entry Point

The main script to start training. It:
- Logs the training configuration (device, batch size, learning rate, etc.)
- Builds the trainer via `build_trainer()`
- Resumes from the latest checkpoint if available
- Runs the training loop with OOM error handling
- Performs aggressive memory cleanup in a `finally` block

#### [`model.py`](model.py) — `MiniT5` Model

The top-level model class that composes the encoder, decoder, and output projection.

- **`__init__`**: Creates a `T5Encoder`, a `T5Decoder`, and a final `nn.Linear` output projection.
- **`gradient_checkpointing_enable/disable`**: Toggles gradient checkpointing for memory-efficient training.
- **`generate_causal_mask`**: Creates a lower-triangular mask for decoder self-attention (prevents attending to future tokens).
- **`forward`**: Runs the encoder, then the decoder, then projects to vocabulary logits. Auto-generates the causal mask if not provided.

#### [`encoder.py`](encoder.py) — `T5Encoder`

The encoder component of the model.

- **`__init__`**: Creates token embedding, positional encoding, a stack of `EncoderBlock` layers, and a final `LayerNorm`.
- **`forward`**: Embeds input tokens, adds positional encoding, passes through all encoder blocks, applies final layer norm.

#### [`decoder.py`](decoder.py) — `T5Decoder`

The decoder component of the model.

- **`__init__`**: Creates token embedding, positional encoding, a stack of `DecoderBlock` layers, and a final `LayerNorm`.
- **`forward`**: Embeds decoder input tokens, adds positional encoding, passes through all decoder blocks (with encoder output for cross-attention), applies final layer norm.

#### [`config.py`](config.py) — `T5Config`

Configuration class for MiniT5. Inherits hardware-auto-detected settings from the shared `ModelConfig` and overrides T5-specific paths and settings.

- Dataset paths: `TRAIN_CSV_PATH`, `VAL_CSV_PATH`
- Model architecture: `EMBEDDING_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `MAX_LENGTH`
- Training settings: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`
- Memory optimization: `USE_MIXED_PRECISION`, `USE_GRADIENT_CHECKPOINTING`, `GRADIENT_CLIP_NORM`
- Checkpoint settings: `CHECKPOINT_DIR`, `TOKENIZER_PATH`
- TensorBoard settings: `TENSORBOARD_LOG_DIR`

#### [`predict.py`](predict.py) — Inference Script

A script for running trained MiniT5 models for translation.

- **`load_best_model`**: Loads the tokenizer and best model checkpoint.
- **`T5Translator`**: A wrapper class that handles the full inference pipeline:
  - Encodes the input text
  - Generates an encoder padding mask
  - Runs autoregressive decoding: starts with BOS token, repeatedly calls the model, takes the last logit, and appends the predicted token until EOS is reached or max length is hit.
  - Uses **greedy decoding** (argmax) for simplicity.
- **`main`**: Interactive CLI for English-to-Farsi translation.

#### [`checkpoint.py`](checkpoint.py) — Legacy Checkpoint Manager

A simpler checkpoint manager (not used by the current trainer, which uses the shared `common.training.checkpoint.CheckpointManager`). Kept for reference.

### `modules/` — Model Building Blocks

#### [`modules/embedding.py`](modules/embedding.py) — `TokenEmbedding`

A wrapper around `nn.Embedding` that scales the output by `sqrt(embedding_dim)`. This scaling factor helps stabilize training by keeping the variance of the embeddings manageable.

#### [`modules/positional_encoding.py`](modules/positional_encoding.py) — `PositionalEncoding`

Implements **sinusoidal positional encoding** as described in the original "Attention Is All You Need" paper.

- Pre-computes a positional encoding matrix using sine (even indices) and cosine (odd indices) functions.
- Registered as a buffer so it moves with the model to the correct device.
- Added to token embeddings in the forward pass, with dropout applied.

#### [`modules/multi_head_attention.py`](modules/multi_head_attention.py) — `MultiHeadAttention`

The core attention mechanism.

- Projects input into Query (Q), Key (K), and Value (V) vectors via separate linear layers.
- Splits Q, K, V into `num_heads` chunks and reshapes to `(batch, num_heads, seq_len, head_dim)`.
- Computes scaled dot-product attention: `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
- Applies an optional mask (for padding or causal masking).
- Combines heads back and projects to the original dimension.

#### [`modules/cross_attention.py`](modules/cross_attention.py) — `CrossAttention`

A thin wrapper around `MultiHeadAttention` for the decoder's cross-attention sub-layer.

- Queries come from the decoder states.
- Keys and Values come from the encoder output.
- This allows the decoder to "look up" information from the encoded input sequence.

#### [`modules/encoder_block.py`](modules/encoder_block.py) — `EncoderBlock`

A single encoder layer.

1. **Self-Attention**: Multi-head self-attention over the input sequence, with optional padding mask.
2. **Residual + LayerNorm**: `x = LayerNorm(x + Attention(x))`
3. **Feed-Forward**: Position-wise FFN (Linear → GELU → Linear).
4. **Residual + LayerNorm**: `x = LayerNorm(x + FFN(x))`

Supports **gradient checkpointing** to trade compute for memory during training.

#### [`modules/decoder_block.py`](modules/decoder_block.py) — `DecoderBlock`

A single decoder layer with three sub-layers:

1. **Masked Self-Attention**: Decoder attends to previous decoder tokens only (causal mask).
2. **Residual + LayerNorm**: `x = LayerNorm(x + SelfAttention(x))`
3. **Cross-Attention**: Decoder attends to encoder output.
4. **Residual + LayerNorm**: `x = LayerNorm(x + CrossAttention(x, encoder_output))`
5. **Feed-Forward**: Position-wise FFN.
6. **Residual + LayerNorm**: `x = LayerNorm(x + FFN(x))`

Also supports gradient checkpointing.

#### [`modules/feed_forward.py`](modules/feed_forward.py) — `FeedForward`

A position-wise feed-forward network: `Linear(embedding_dim, ff_hidden_dim) → GELU → Dropout → Linear(ff_hidden_dim, embedding_dim) → Dropout`. The 4× expansion ratio is standard in transformer architectures.

#### [`modules/layer_norm.py`](modules/layer_norm.py) — `LayerNorm`

A custom implementation of layer normalization (not using `nn.LayerNorm` directly). Normalizes across the last dimension, then applies learnable scale (`gamma`) and shift (`beta`) parameters.

#### [`modules/tokenizer.py`](modules/tokenizer.py) — `CharacterTokenizer`

A character-level tokenizer for the translation task.

- **Special tokens**: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- **`fit(texts)`**: Builds vocabulary from a list of texts by collecting all unique characters.
- **`encode(text)`**: Converts text to token IDs, adding BOS/EOS if requested.
- **`decode(ids)`**: Converts token IDs back to text, optionally skipping special tokens.
- **`save/load`**: Persists the vocabulary as JSON.

#### [`modules/dataset.py`](modules/dataset.py) — `TranslationDataset`

A PyTorch `Dataset` for the English-to-Farsi translation task.

- Reads a CSV file with `source` and `target` columns.
- Tokenizes both source and target text.
- Creates decoder inputs by shifting the target sequence right by one (standard teacher forcing).
- Pads or truncates sequences to `max_length`.
- Returns a dictionary with `encoder_input_ids`, `decoder_input_ids`, and `labels`.

### `train/` — Training Loop

#### [`train/trainer.py`](train/trainer.py) — `T5Trainer` & `build_trainer()`

- **`build_trainer()`**: Factory function that assembles all training components:
  - Loads and fits the tokenizer on train + validation data.
  - Creates `TranslationDataset` and `DataLoader` instances.
  - Builds the `MiniT5` model.
  - Sets up `CrossEntropyLoss` (ignoring padding tokens), `AdamW` optimizer, `CheckpointManager`, `EarlyStopping`, and `TensorBoardLogger`.
  - Returns a `T5Trainer` instance.

- **`T5Trainer`**: Orchestrates the training loop:
  - Iterates over epochs, calling `train_one_epoch()` and `validate_one_epoch()`.
  - Logs metrics to TensorBoard.
  - Saves checkpoints and tracks the best model.
  - Checks early stopping.
  - Performs aggressive memory cleanup between phases.

#### [`train/train_one_epoch.py`](train/train_one_epoch.py) — `train_one_epoch()`

Runs one epoch of training with several memory-optimization techniques:

- **Gradient Accumulation**: Accumulates gradients over multiple batches before stepping the optimizer, simulating a larger effective batch size.
- **Mixed Precision**: Uses `torch.cuda.amp.GradScaler` and `autocast` for FP16 training, reducing memory usage and speeding up training on compatible GPUs.
- **Gradient Clipping**: Clips gradients to prevent exploding gradients.
- **Aggressive Memory Cleanup**: Deletes tensors after each batch and periodically clears the CUDA cache.

#### [`train/validate_one_epoch.py`](train/validate_one_epoch.py) — `validate_one_epoch()`

Runs one epoch of validation:

- Uses `torch.no_grad()` to disable gradient computation.
- Computes average validation loss.
- Performs the same memory cleanup as training.

---

## Training & Inference

### Training

```bash
python -m mini_t5.main_train
```

This will:
1. Load the training and validation CSVs.
2. Build the character-level tokenizer.
3. Create the model, optimizer, and data loaders.
4. Train for up to 50 epochs with early stopping.
5. Save checkpoints to `mini_t5/checkpoints/`.
6. Log metrics to `mini_t5/runs/` (TensorBoard).

### Inference

```bash
python -m mini_t5.predict
```

This launches an interactive CLI where you can type English sentences and get Farsi translations.

### Monitoring

```bash
tensorboard --logdir mini_t5/runs
```

---

## Configuration

MiniT5 uses a two-level configuration system:

1. **`common/configs/model_config.py`** — `ModelConfig`: Auto-detects optimal settings based on available hardware (GPU/CPU memory). Provides defaults for batch size, gradient accumulation, mixed precision, gradient checkpointing, etc.

2. **`mini_t5/config.py`** — `T5Config`: Inherits from `ModelConfig` and overrides T5-specific settings such as dataset paths, checkpoint directory, and TensorBoard log directory.

This design allows all models in the project (mini_gpt, mini_bert, mini_t5) to share the same hardware-adaptive configuration logic while maintaining their own specific settings.
