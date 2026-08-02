# MiniGPT — A Complete Teaching Guide

> **For students who want to understand and implement a GPT-style language model from scratch.**

This guide walks through every file, every class, and every design decision in the `mini_gpt/` project. By the end, you will understand not just *what* the code does, but *why* each piece exists and *how* to build it yourself.

---

## Table of Contents

1. [What is MiniGPT?](#1-what-is-minigpt)
2. [How the Project is Organized](#2-how-the-project-is-organized)
3. [How It Works — The Big Picture](#3-how-it-works--the-big-picture)
4. [How to Run It](#4-how-to-run-it)
5. [File-by-File Deep Dive](#5-file-by-file-deep-dive)
6. [Architecture Decisions — Why Things Are Done This Way](#6-architecture-decisions--why-things-are-done-this-way)
7. [Implementing It From Scratch — Step by Step](#7-implementing-it-from-scratch--step-by-step)

---

## 1. What is MiniGPT?

MiniGPT is a **minimal, from-scratch implementation of a GPT (Generative Pre-trained Transformer)** — specifically a **decoder-only autoregressive language model**.

Given a sequence of token IDs, it predicts the probability distribution over the vocabulary for the *next* token. This is called **causal language modeling**.

Key characteristics:
- **Decoder-only**: Uses only the decoder stack (no encoder). This is the GPT architecture.
- **Autoregressive**: Generates one token at a time, conditioning each new token on all previous tokens.
- **Character-level**: The tokenizer operates at the character level, meaning every single character is a token. This makes it simple but limits the vocabulary size.
- **Small**: Designed to run on consumer hardware (even CPU-only machines) with aggressive memory optimizations.

---

## 2. How the Project is Organized

```
mini_gpt/
├── config.py          # Configuration (re-exports shared ModelConfig)
├── model.py           # The MiniGPT neural network model
├── train.py           # Entry point: sets up everything and starts training
├── trainer.py         # Training loop orchestrator
├── train_one_epoch.py # One epoch of training logic
├── validate_one_epoch.py # One epoch of validation logic
├── generate.py        # Inference script: generate text from a prompt
├── checkpoints/       # Saved model checkpoints (created at runtime)
└── runs/              # TensorBoard logs (created at runtime)
```

The project also depends heavily on shared code in `common/`:
- `common/nn/` — neural network layers and blocks
- `common/data/` — dataset, tokenizer, vocabulary
- `common/configs/` — configuration classes
- `common/training/` — loss, checkpointing, early stopping, logging

---

## 3. How It Works — The Big Picture

### 3.1 The Forward Pass

When MiniGPT processes a batch of token sequences, here's what happens inside the model:

```
input_ids (batch, seq_len)
    │
    ▼
┌─────────────────────────┐
│  InputEmbedding          │  ← Token embedding + Positional embedding
│  (vocab_size → dim)      │     Adds information about what each token IS
│                          │     and WHERE it is in the sequence
└────────────┬────────────┘
             │  (batch, seq_len, embedding_dim)
             ▼
┌─────────────────────────┐
│  DecoderBlock × N        │  ← N identical transformer blocks
│  ┌─────────────────────┐│
│  │ LayerNorm → Attention││  ← Self-attention with causal mask
│  │ + Residual connection││
│  │ LayerNorm → FeedForward││ ← Two-layer MLP with GELU
│  │ + Residual connection││
│  └─────────────────────┘│
└────────────┬────────────┘
             │  (batch, seq_len, embedding_dim)
             ▼
┌─────────────────────────┐
│  LayerNorm               │  ← Final normalization
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  lm_head (Linear)        │  ← Projects back to vocabulary size
│  (embedding_dim → vocab) │     Produces logits for each token position
└────────────┬────────────┘
             │
             ▼
logits (batch, seq_len, vocab_size)
    │
    ▼  (if targets provided)
loss = CrossEntropyLoss(logits, targets)
```

### 3.2 The Training Loop

1. Load pre-processed binary data (`train.bin`, `validation.bin`) via `TextDataset`.
2. Create `DataLoader` instances that yield `(x, y)` batches where:
   - `x` = input token IDs (the first `block_size` tokens)
   - `y` = target token IDs (the next token after each position in `x`)
3. For each batch:
   - Forward pass → get logits and loss
   - Backward pass → compute gradients
   - Optimizer step → update weights
4. After each epoch, validate on the held-out set.
5. Save checkpoints and stop early if validation loss doesn't improve.

### 3.3 Text Generation

1. Encode a prompt string into token IDs.
2. For each new token to generate:
   - Run the model on the current context (truncated to `block_size`).
   - Take the logits for the *last* position only.
   - Apply temperature scaling to control randomness.
   - Either sample from the distribution or take the argmax.
   - Append the new token ID to the context.
3. Decode the generated token IDs back to text.

---

## 4. How to Run It

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare the dataset (run once)
python data_pipeline.py
# Or equivalently:
python -m data_pipeline
```

This reads `common/data/raw/train.csv` and `common/data/raw/validation.csv`, builds a character vocabulary, and writes:
- `common/data/processed/vocab.json` — the vocabulary mapping
- `common/data/processed/train.bin` — tokenized training data (binary)
- `common/data/processed/validation.bin` — tokenized validation data (binary)

### Training

```bash
# Run the full pipeline (data + training)
python main_gpt.py

# Or run training directly (assuming data is already prepared)
python -m mini_gpt.train
```

Training will:
- Auto-detect your hardware (GPU/CPU) and configure batch size, precision, etc.
- Train for up to 10 epochs (or stop early if validation loss doesn't improve).
- Use fixed step counts per epoch (`STEPS_PER_EPOCH=5000`, `VAL_STEPS=200`) for deterministic epoch sizes.
- Save checkpoints to `mini_gpt/checkpoints/`.
- Log metrics to `mini_gpt/runs/` for TensorBoard.

### Text Generation

```bash
python -m mini_gpt.generate
```

This loads the best checkpoint and generates text starting from the prompt `"Once upon a time"`.

### Testing Individual Components

```bash
python test_attention.py      # Test multi-head attention
python test_decoder_block.py  # Test a single decoder block
python test_embedding.py      # Test input embedding
python test_gpt.py            # Test the full MiniGPT model
python data_pipeline_test.py  # Test the dataset and dataloader
```

---

## 5. File-by-File Deep Dive

### 5.1 `mini_gpt/config.py`

```python
from common.configs.model_config import ModelConfig
GPTConfig = ModelConfig
```

**What it does**: Re-exports `ModelConfig` as `GPTConfig` for backward compatibility.

**Why it exists**: The `mini_gpt` package used to have its own config class. When the shared `ModelConfig` was introduced with auto-detection, this file preserves the old import path so any code referencing `GPTConfig` still works.

**Key takeaway**: This is a compatibility shim. In a real project, you might eventually remove it and update all imports to use `ModelConfig` directly.

---

### 5.2 `mini_gpt/model.py` — `MiniGPT`

This is the heart of the project. Let's break it down section by section.

#### Constructor

```python
def __init__(
    self,
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float = 0.1,
    use_gradient_checkpointing: bool = False,
):
```

**Parameters explained**:
| Parameter | Meaning |
|-----------|---------|
| `vocab_size` | How many unique tokens exist (e.g., number of characters + special tokens) |
| `block_size` | Maximum sequence length the model can process. Longer sequences are truncated. |
| `embedding_dim` | The size of each token's vector representation. Larger = more expressive but more memory. |
| `num_heads` | Number of attention heads in multi-head attention. Must divide `embedding_dim` evenly. |
| `num_layers` | Number of decoder blocks to stack. More layers = deeper model = more capacity but slower. |
| `dropout` | Probability of zeroing out attention weights and FFN activations during training for regularization. |
| `use_gradient_checkpointing` | If True, recomputes activations during backward pass instead of storing them. Saves memory at the cost of compute. |

**Why these defaults?**
- `embedding_dim=128`, `num_heads=4`, `num_layers=4` — small enough to train quickly on CPU/GPU with limited memory.
- `block_size=128` — short sequences keep memory usage manageable. The model can only "see" 128 tokens at a time.
- `dropout=0.1` — standard regularization for transformers.

#### Embedding Layer

```python
self.embedding = InputEmbedding(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
)
```

**Why `InputEmbedding`?** It combines token embeddings and positional embeddings in one module. The model needs to know both *what* each token is and *where* it is in the sequence. Without positional embeddings, the model would be permutation-invariant and unable to distinguish "the cat sat" from "sat cat the".

#### Decoder Blocks

```python
self.blocks = nn.ModuleList(
    [
        DecoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        for _ in range(num_layers)
    ]
)
```

**Why `nn.ModuleList`?** It's the correct PyTorch container for a list of submodules. Unlike a plain Python list, `nn.ModuleList` registers each module so that parameters are included in `model.parameters()` and are properly moved to GPU/CPU with `model.to(device)`.

**Why a loop instead of hardcoding?** The number of layers is a hyperparameter. Using a loop makes it easy to experiment with different depths.

**Why `DecoderBlock`?** This is the GPT architecture — a stack of identical decoder blocks, each containing:
1. Multi-head self-attention with causal masking
2. Feed-forward network
3. Residual connections around each sub-layer
4. Layer normalization (pre-norm style)

#### Final Layers

```python
self.norm = LayerNorm(embedding_dim)
self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
```

**Why `LayerNorm` before the head?** The final layer norm stabilizes the output distribution before the projection to vocabulary size. This is part of the pre-norm residual stream design.

**Why `bias=False` in `lm_head`?** The bias term in the final linear layer is redundant because `LayerNorm` already learns a bias (its `beta` parameter). Removing it reduces parameter count with no loss in expressiveness. This is a standard GPT design choice.

#### Weight Initialization

```python
self.apply(self._init_weights)
```

```python
def _init_weights(self, module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Why this initialization?**
- `N(0, 0.02)` for linear layers and embeddings — this is the initialization used in the original GPT-2 paper. It keeps initial activations small and prevents them from exploding or vanishing at the start of training.
- Zero-initializing biases ensures they start as identity mappings, so the network starts in a "neutral" state.
- `self.apply()` recursively applies `_init_weights` to every submodule, so all layers get initialized correctly.

#### Gradient Checkpointing Methods

```python
def gradient_checkpointing_enable(self) -> None:
    self._use_gradient_checkpointing = True
    for block in self.blocks:
        block.use_gradient_checkpointing = True

def gradient_checkpointing_disable(self) -> None:
    self._use_gradient_checkpointing = False
    for block in self.blocks:
        block.use_gradient_checkpointing = False
```

**Why these methods?** Gradient checkpointing is a memory optimization that trades compute for memory. Instead of storing all intermediate activations during the forward pass (which requires a lot of GPU RAM), it recomputes them during the backward pass. This allows training larger models or larger batch sizes on limited hardware.

**Why toggle methods instead of a constructor flag?** They allow enabling/disabling checkpointing *after* the model is created, which is useful for experimentation without recreating the model.

#### Forward Method

```python
def forward(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = self.embedding(input_ids)
    for block in self.blocks:
        x = block(x)
    x = self.norm(x)
    logits = self.lm_head(x)
    loss = None
    if targets is not None:
        batch_size, sequence_length, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.reshape(batch_size * sequence_length, vocab_size),
            targets.reshape(batch_size * sequence_length),
        )
    return logits, loss
```

**Step-by-step explanation**:
1. `self.embedding(input_ids)` — converts token IDs to vectors of shape `(batch, seq_len, embedding_dim)`.
2. Each `DecoderBlock` processes the sequence, maintaining the same shape.
3. `self.norm(x)` — final layer normalization.
4. `self.lm_head(x)` — projects each position's vector to a logits vector of size `vocab_size`. Output shape: `(batch, seq_len, vocab_size)`.
5. If `targets` is provided, compute cross-entropy loss by flattening the batch and sequence dimensions.

**Why flatten before computing loss?** `nn.CrossEntropyLoss` expects input of shape `(N, C)` where N is the number of samples and C is the number of classes. By reshaping `(batch*seq_len, vocab_size)`, we treat each token position as an independent classification problem.

**Why return `loss=None` when targets is not provided?** During inference (text generation), we don't need a loss — we only need logits to predict the next token. Returning `None` makes this explicit.

#### Generate Method

```python
@torch.inference_mode()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> torch.Tensor:
```

**Step-by-step explanation**:
1. `@torch.inference_mode()` — disables gradient computation and sets the model to inference mode. This saves memory and speeds up computation.
2. `was_training = self.training; self.eval()` — saves the training mode and switches to eval mode (important for dropout and batch norm behavior).
3. The loop runs `max_new_tokens` times:
   - `context = input_ids[:, -self.block_size:]` — truncates the context to `block_size` tokens. The model can only attend to `block_size` positions due to the fixed positional embedding table.
   - `logits, _ = self(context)` — runs the forward pass. We discard the loss (no targets during generation).
   - `logits = logits[:, -1, :]` — takes only the logits for the *last* token position, because we want to predict the *next* token.
   - `logits = logits / temperature` — scales the logits. Lower temperature = more confident (greedy-like), higher temperature = more uniform (more random).
   - `probabilities = torch.softmax(logits, dim=-1)` — converts logits to a probability distribution.
   - If `do_sample=True`: `torch.multinomial(probabilities, num_samples=1)` — samples from the distribution (introduces randomness).
   - If `do_sample=False`: `torch.argmax(probabilities, dim=-1, keepdim=True)` — takes the most likely token (greedy decoding).
   - `input_ids = torch.cat((input_ids, next_token), dim=1)` — appends the new token to the context.
4. `self.train(was_training)` — restores the original training mode.
5. Returns the full sequence including the generated tokens.

**Why truncate to `block_size`?** The positional embedding table only has `block_size` entries (indices 0 to `block_size-1`). If the sequence is longer, we can't look up positional embeddings for positions beyond `block_size`. Truncation is a practical limitation of fixed-position embeddings.

**Why `temperature`?** Temperature controls the sharpness of the probability distribution:
- `temperature → 0`: the distribution becomes a one-hot vector (argmax, deterministic).
- `temperature = 1.0`: the raw softmax distribution.
- `temperature → ∞`: the distribution becomes uniform (maximum randomness).

**Why save and restore training mode?** Dropout behaves differently in train vs. eval mode. During generation, we want dropout disabled (all neurons active). Restoring the original mode ensures the model continues training correctly after generation.

---

### 5.3 `mini_gpt/train.py` — Training Entry Point

This file is the main entry point for training. It sets up every component and starts the training loop.

#### Building the Trainer

```python
def build_trainer() -> Trainer:
```

**Step 1: Load datasets**

```python
train_dataset = TextDataset(
    Path("common/data/processed/train.bin"),
    ModelConfig.BLOCK_SIZE,
)
val_dataset = TextDataset(
    Path("common/data/processed/validation.bin"),
    ModelConfig.BLOCK_SIZE,
)
```

**Why `TextDataset`?** It memory-maps the binary files, so only the small slices needed for each batch are actually read into memory. This allows training on datasets much larger than available RAM.

**Why `block_size`?** The dataset creates input-target pairs of length `block_size`. Each sample is a sliding window of `block_size` tokens from the binary file. With the new `stride` parameter, windows can overlap for more granular training samples.

**Step 2: Create DataLoaders with fixed step counts**

```python
train_sampler = RandomSampler(
    train_dataset,
    replacement=True,
    num_samples=ModelConfig.STEPS_PER_EPOCH * ModelConfig.BATCH_SIZE,
)
```

**Why `RandomSampler` with `replacement=True`?** With replacement, the sampler can draw the same index multiple times per epoch. This ensures we always get exactly `STEPS_PER_EPOCH * BATCH_SIZE` samples, making each epoch have a fixed number of batches regardless of dataset size.

**Why fixed step counts (`STEPS_PER_EPOCH`, `VAL_STEPS`)?** Decoupling epoch size from dataset length provides:
- **Deterministic training**: Each epoch has the same number of batches, making learning rate schedules and progress tracking predictable.
- **Reproducibility**: Training runs are consistent across different dataset sizes.
- **Flexibility**: You can train for a fixed number of steps without worrying about dataset size variations.

**Validation sampler:**

```python
val_sampler = RandomSampler(
    val_dataset,
    replacement=True,
    num_samples=ModelConfig.VAL_STEPS * ModelConfig.BATCH_SIZE,
)
```

**Why a separate validation sampler?** Previously, validation used `shuffle=False` with the full dataset. Now it also uses a fixed-step `RandomSampler` for consistency and deterministic epoch sizes.

**Ultra-low-memory DataLoader settings:**

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=ModelConfig.BATCH_SIZE,
    sampler=train_sampler,
    pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available(),
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=False,
    drop_last=True,
    multiprocessing_context=None,
)
```

Each of these settings is critical for low-memory operation:
- **`num_workers=0`**: Each DataLoader worker maps the entire file into memory. With multiple workers, you'd multiply memory usage by the number of workers. Single process avoids this.
- **`pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available()`**: Dynamic pin memory — only enables pinned memory when both the config says to and a GPU is actually available. On CPU-only systems, this saves RAM.
- **`prefetch_factor=None`**: No prefetching with a single worker.
- **`persistent_workers=False`**: Don't keep worker processes alive between epochs (saves memory).
- **`drop_last=True`**: Drop the last incomplete batch. This avoids variable batch sizes that complicate gradient accumulation.
- **`multiprocessing_context=None`**: Uses the default fork method instead of spawn, which is lighter.

**Step 3: Load vocabulary and create model**

```python
vocabulary = Vocabulary.load(ModelConfig.VOCAB_PATH)
model = MiniGPT(
    vocab_size=vocabulary.size,
    block_size=ModelConfig.BLOCK_SIZE,
    embedding_dim=ModelConfig.EMBEDDING_DIM,
    num_heads=ModelConfig.NUM_HEADS,
    num_layers=ModelConfig.NUM_LAYERS,
    dropout=ModelConfig.DROPOUT,
    use_gradient_checkpointing=ModelConfig.USE_GRADIENT_CHECKPOINTING,
)
```

**Why load vocabulary separately?** The vocabulary is needed to know `vocab_size` (the number of unique tokens). It's also needed for tokenization during generation.

**Step 4: Set up optimizer, loss, and training utilities**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=ModelConfig.LEARNING_RATE)
criterion = LanguageModelLoss()
checkpoint_manager = CheckpointManager(...)
early_stopping = EarlyStopping(...)
tb_logger = TensorBoardLogger(...)
```

**Why `AdamW` instead of `Adam`?** AdamW decouples weight decay from the adaptive learning rate update. This leads to better generalization and is the standard optimizer for transformer training (used in GPT-2, BERT, etc.).

**Why `LanguageModelLoss` instead of raw `nn.CrossEntropyLoss`?** It's a thin wrapper that handles the reshaping automatically. It makes the training code cleaner and encapsulates the loss computation logic.

**Step 5: Return the Trainer**

```python
return Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    device=ModelConfig.DEVICE,
    checkpoint_manager=checkpoint_manager,
    early_stopping=early_stopping,
    tb_logger=tb_logger,
    epochs=ModelConfig.EPOCHS,
)
```

#### The `main()` Function

```python
def main():
    # Log configuration
    # Build trainer
    # Try to resume from latest checkpoint
    # Run training with OOM error handling
```

**Configuration logging:**

The log header is now `"Training Configuration"` (previously `"ULTRA LOW-MEMORY Training Configuration"`), reflecting that the configuration is now more general and not exclusively focused on ultra-low-memory scenarios. The batch size is now uniformly 32 across all hardware tiers.

**Why resume from checkpoint?** Training can be interrupted (system crash, manual stop, OOM). Resuming from the latest checkpoint means you don't lose progress.

**Why OOM error handling?** Out-of-memory errors are common when training transformers. The error handler provides actionable suggestions (reduce batch size, enable gradient checkpointing, etc.) instead of just crashing with a cryptic PyTorch error.

**Why GPU memory logging?** The `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.memory_allocated()` calls help monitor GPU memory usage at the start of training, which is useful for debugging memory issues.

---

### 5.4 `mini_gpt/trainer.py` — `Trainer`

The `Trainer` class orchestrates the training loop.

#### Constructor

```python
def __init__(self, model, optimizer, criterion, train_loader, val_loader,
             device, checkpoint_manager, early_stopping, tb_logger, epochs):
```

Each parameter is stored as an instance attribute. The trainer doesn't own these objects — it borrows them and coordinates their interaction.

#### Memory Cleanup

```python
def _cleanup_memory(self, clear_cuda: bool = True) -> None:
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        gc.collect()
```

**Why aggressive cleanup?** PyTorch's memory allocator holds onto GPU memory even after tensors are deleted. `torch.cuda.empty_cache()` releases unused memory back to the GPU. Running `gc.collect()` three times ensures Python's garbage collector finds and frees all unreferenced objects.

#### Training Loop

```python
def train(self, start_epoch: int = 0) -> None:
    for epoch in range(start_epoch, self.epochs):
        # Train
        train_loss = train_one_epoch(...)
        # Cleanup
        self._cleanup_memory()
        # Validate
        val_loss = validate_one_epoch(...)
        # Cleanup
        self._cleanup_memory()
        # Log metrics
        # Save checkpoint
        # Check early stopping
        # Extra cleanup
    # Final cleanup
    del self.train_loader
    del self.val_loader
    self._cleanup_memory()
    self.tb_logger.close()
```

**Why cleanup between training and validation?** Training creates intermediate tensors (activations, gradients) that consume memory. Before validation, we need to free that memory so validation can run efficiently.

**Why delete loaders at the end?** `DataLoader` with `num_workers=0` still holds a reference to the dataset, which holds the memory-mapped file. Deleting the loaders and cleaning up ensures the file mappings are released.

**Why close the TensorBoard writer?** It flushes any remaining logs and releases file handles. Not closing it can lead to incomplete log files.

---

### 5.5 `mini_gpt/train_one_epoch.py` — `train_one_epoch()`

This function runs a single epoch of training with several memory and stability optimizations.

#### Mixed Precision Training

```python
scaler = GradScaler() if use_mixed_precision else None
```

**What is mixed precision?** It uses 16-bit floating point (FP16/BF16) for forward and backward passes instead of 32-bit (FP32). This halves memory usage and speeds up computation on modern GPUs.

**Why `GradScaler`?** FP16 has a small dynamic range. Gradients can underflow to zero. The scaler multiplies the loss by a scale factor before backpropagation, so gradients stay in a representable range. After unscaling, gradients are checked for overflow and the scale is adjusted dynamically.

#### Gradient Accumulation

```python
loss = loss / gradient_accumulation_steps
```

**What is gradient accumulation?** Instead of updating weights after every batch, we accumulate gradients over `gradient_accumulation_steps` batches and then update once. This effectively increases the batch size without using more memory.

**Why divide the loss?** Since we're adding gradients from multiple forward passes, we need to average them. Dividing the loss by the number of accumulation steps ensures the final gradient is the average, not the sum.

#### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
```

**What does this do?** It rescales all gradients so that their total norm doesn't exceed `gradient_clip_norm`. This prevents "exploding gradients" — a common problem in training deep networks where gradients grow exponentially large, causing numerical overflow.

**Why unscale before clipping?** When using mixed precision, gradients are stored in the scaled space. `scaler.unscale_(optimizer)` converts them back to the original scale before clipping.

#### Memory Cleanup Per Batch

```python
del x, y, logits, loss
```

**Why delete tensors after each batch?** Python's reference counting would eventually free them, but in a tight loop, the garbage collector might not run fast enough. Explicit deletion ensures memory is freed immediately, preventing accumulation across batches.

#### Handling Remaining Gradients

```python
if num_batches % gradient_accumulation_steps != 0:
    # Step optimizer with remaining accumulated gradients
```

**Why?** If the total number of batches isn't divisible by `gradient_accumulation_steps`, the last few batches have accumulated gradients but haven't triggered an optimizer step yet. This code ensures those gradients are still applied.

---

### 5.6 `mini_gpt/validate_one_epoch.py` — `validate_one_epoch()`

This function runs one epoch of validation. It's simpler than training because:
- No optimizer step
- No gradient accumulation
- No mixed precision (to keep it simple and deterministic)
- `torch.no_grad()` context — disables gradient computation entirely, saving memory

**Why `torch.no_grad()`?** During validation, we don't need gradients. Disabling them saves significant memory (the GPU doesn't need to store intermediate values for backprop) and speeds up computation.

---

### 5.7 `mini_gpt/generate.py` — Text Generation Script

This script demonstrates how to use a trained model for inference.

#### Loading the Model

```python
model = MiniGPT(
    vocab_size=vocabulary_size,
    block_size=DataConfig.BLOCK_SIZE,
    embedding_dim=ModelConfig.EMBEDDING_DIM,
    num_heads=ModelConfig.NUM_HEADS,
    num_layers=ModelConfig.NUM_LAYERS,
    dropout=ModelConfig.DROPOUT,
)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
```

**Why `map_location=device`?** When loading a checkpoint saved on GPU, `torch.load` tries to map tensors to GPU by default. If you're running on CPU, this fails. `map_location=device` ensures tensors are loaded directly onto the correct device.

**Why `model.eval()`?** Switches the model to evaluation mode, which disables dropout and other training-only behaviors.

#### Generation Parameters

```python
generated = model.generate(
    input_tensor,
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True,
)
```

**Why `temperature=0.8`?** Slightly lower than 1.0 makes the model more confident and less random, producing more coherent text while still allowing some diversity.

**Why `do_sample=True`?** Sampling from the distribution produces more varied and interesting text than always picking the most likely token (greedy decoding). Greedy decoding tends to produce repetitive, generic text.

---

## 6. Architecture Decisions — Why Things Are Done This Way

### 6.1 Why a Decoder-Only Architecture?

GPT is a decoder-only model. Unlike BERT (encoder-only) or T5 (encoder-decoder), GPT only uses the "generate" part of the transformer. This is the right choice for:
- **Autoregressive text generation**: You want to predict the next token given all previous tokens.
- **Simplicity**: Fewer components to implement and debug.
- **Scaling**: Decoder-only models scale well to very large sizes (GPT-2, GPT-3, GPT-4 are all decoder-only).

### 6.2 Why Pre-Norm (LayerNorm Before Attention/FFN)?

The original GPT-2 used post-norm (LayerNorm after the sub-layer). Modern implementations (including this one) use pre-norm because:
- It stabilizes training, especially with deep networks.
- It eliminates the need for a learning rate warmup phase in many cases.
- Residual connections after normalization are more stable.

### 6.3 Why Causal Masking?

In autoregressive language modeling, each token should only attend to *previous* tokens, not future ones. The causal mask sets all future positions to `-inf` before the softmax, so their attention weights become zero.

Without causal masking, the model could "cheat" by looking at future tokens during training, which would make it impossible to use the model for actual generation (where you don't have future tokens yet).

### 6.4 Why `nn.Linear(embedding_dim, embedding_dim * 3)` for QKV Projection?

Instead of three separate linear layers for Q, K, and V, we project all three at once with a single larger linear layer and then split the result. This is more efficient because:
- One matrix multiplication instead of three.
- Better memory locality and cache utilization.
- The fused operation is optimized in PyTorch's backend.

### 6.5 Why `bias=False` in `lm_head`?

The final linear layer projects from `embedding_dim` to `vocab_size`. Since `LayerNorm` already has a learnable bias (`beta`), adding another bias in the linear layer is redundant. Removing it reduces the parameter count by `vocab_size` with no loss in expressiveness.

### 6.6 Why `nn.ModuleList` Instead of a Python List?

`nn.ModuleList` is a PyTorch container that:
- Registers each module as a submodule of the parent.
- Ensures parameters are included in `model.parameters()`.
- Properly handles device placement (`model.to(device)`).
- Saves and loads correctly with `state_dict()`.

A plain Python list would not do any of these things, and the model's parameters would be invisible to the optimizer and the state dict.

### 6.7 Why Memory-Mapped Files for the Dataset?

`np.memmap` creates a memory-mapped array that reads data from disk on demand. This means:
- The entire dataset doesn't need to fit in RAM.
- Only the small slices accessed by each batch are actually loaded.
- The OS handles caching frequently accessed pages automatically.

For large datasets, this is essential. A 10GB dataset can be "loaded" instantly with memory mapping, whereas loading it into a Python list would require 10GB of RAM.

### 6.8 Why `np.uint16` for Token IDs?

Token IDs are non-negative integers. With a vocabulary of up to 65,535 tokens, `uint16` (unsigned 16-bit integer) is sufficient and uses half the memory of `int32`. This is important for large datasets where memory efficiency matters.

### 6.9 Why Gradient Checkpointing?

During the forward pass, PyTorch stores all intermediate activations so they can be used during the backward pass. For deep networks, these activations consume a lot of GPU memory.

Gradient checkpointing trades compute for memory: instead of storing activations, it recomputes them during the backward pass. This reduces memory usage at the cost of ~30% more computation.

### 6.10 Why `torch.inference_mode()` Instead of `torch.no_grad()`?

`torch.inference_mode()` is a newer, stricter version of `torch.no_grad()`. It:
- Disables gradient computation.
- Also disables version counter tracking for tensors, which further reduces memory overhead.
- Signals to PyTorch that the code is in inference mode, enabling additional optimizations.

### 6.11 Why Fixed Step Counts (`STEPS_PER_EPOCH`, `VAL_STEPS`)?

Previously, epoch size was determined by the dataset length (`len(train_dataset)`). This meant:
- Different dataset sizes → different number of batches per epoch.
- Learning rate schedules that depend on epoch count would behave inconsistently.
- Reproducibility issues across different runs with different data.

With fixed step counts:
- Each epoch has exactly `STEPS_PER_EPOCH` training batches and `VAL_STEPS` validation batches.
- The `RandomSampler` with `replacement=True` ensures this by sampling with replacement.
- Learning rate schedules and progress tracking become predictable and reproducible.
- The dataset can grow or shrink without affecting the training loop structure.

### 6.12 Why Strided Windows in `TextDataset`?

The new `stride` parameter in `TextDataset` allows overlapping windows:
- `stride=block_size` (default): Non-overlapping windows, standard approach.
- `stride < block_size`: Overlapping windows, which increases the number of training samples from the same data and provides more diverse training examples.
- `stride > block_size`: Gaps between windows, which reduces the number of samples but may skip redundant patterns.

This is useful for controlling the effective dataset size and training density.

### 6.13 Why Dynamic `pin_memory`?

`pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available()` ensures that pinned memory is only enabled when:
1. The config says to use it, AND
2. A GPU is actually available.

On CPU-only systems, pinned memory is unnecessary and wastes RAM. This dynamic check prevents that waste.

---

## 7. Implementing It From Scratch — Step by Step

If you want to implement this project from scratch, here's the order you should build things:

### Step 1: Set Up the Project Structure

Create the directory structure:
```
transformers-from-scratch/
├── common/
│   ├── nn/
│   │   ├── layers/
│   │   └── blocks/
│   ├── data/
│   │   ├── readers/
│   │   ├── builders/
│   │   └── preprocessing/
│   ├── configs/
│   ├── training/
│   └── utils/
├── mini_gpt/
└── common/
```

### Step 2: Implement the Core Neural Network Layers

Start with the simplest components and build up:

1. **`TokenEmbedding`** — A wrapper around `nn.Embedding`. This is the simplest layer.
2. **`PositionalEmbedding`** — Another `nn.Embedding` that looks up position indices.
3. **`InputEmbedding`** — Combines token and positional embeddings via addition.
4. **`SelfAttention`** — Single-head attention with causal masking. Implement this carefully; it's the core of the transformer.
5. **`MultiHeadAttention`** — Wraps `SelfAttention` to run multiple heads in parallel.
6. **`FeedForward`** — Two linear layers with GELU activation.
7. **`LayerNorm`** — Thin wrapper around `nn.LayerNorm`.
8. **`DecoderBlock`** — Combines attention, FFN, residuals, and layer norms.

### Step 3: Implement the Model

1. **`MiniGPT`** — Stacks `InputEmbedding` + N `DecoderBlock`s + `LayerNorm` + `lm_head`.
2. Implement `_init_weights()` using the GPT-2 initialization scheme.
3. Implement `forward()` — embedding → blocks → norm → head → (optional loss).
4. Implement `generate()` — the autoregressive loop.

### Step 4: Implement the Data Pipeline

1. **`Vocabulary`** — Bidirectional token↔ID mapping with save/load.
2. **`CharacterVocabularyBuilder`** — Counts characters and builds a vocabulary.
3. **`CharacterTokenizer`** — Encodes/decodes text at the character level.
4. **`DatasetPreparer`** — Builds vocabulary and encodes text to binary files.
5. **`TextDataset`** — Memory-mapped dataset with configurable stride that yields `(x, y)` pairs.

### Step 5: Implement Training Infrastructure

1. **`LanguageModelLoss`** — Cross-entropy with reshaping.
2. **`CheckpointManager`** — Save/load checkpoints.
3. **`EarlyStopping`** — Stop training when validation loss stops improving.
4. **`TensorBoardLogger`** — Log metrics for visualization.
5. **`train_one_epoch()`** — The training loop with gradient accumulation and mixed precision.
6. **`validate_one_epoch()`** — The validation loop.
7. **`Trainer`** — Orchestrates the training loop.

### Step 6: Implement Configuration

1. **`ModelConfig`** — Auto-detects hardware and sets optimal training parameters.
   - Key fields: `STEPS_PER_EPOCH=5000`, `VAL_STEPS=200` for fixed step counts.
   - `BATCH_SIZE=32` uniformly across all hardware tiers.
2. **`DataConfig`** — Legacy data configuration.

### Step 7: Implement the Application Scripts

1. **`mini_gpt/train.py`** — Entry point that sets up everything and starts training.
   - Uses `STEPS_PER_EPOCH` and `VAL_STEPS` for fixed-size epochs.
   - Uses `RandomSampler` for both train and validation with replacement.
   - Dynamic `pin_memory` based on GPU availability.
2. **`mini_gpt/trainer.py`** — Training loop orchestrator that coordinates training and validation phases, performs memory cleanup, logs metrics, saves checkpoints, and checks early stopping.

### Key Principles to Follow

1. **Start small, then scale.** Begin with `embedding_dim=16`, `num_layers=1`, `block_size=16`. Once it works, increase the size.
2. **Verify shapes at every step.** Print tensor shapes after each operation to catch bugs early.
3. **Use memory-mapped files for data.** Don't load everything into RAM.
4. **Implement gradient checkpointing early.** It will save you from OOM errors when you scale up.
5. **Save checkpoints frequently.** You don't want to lose hours of training to a crash.
6. **Use mixed precision on GPU.** It halves memory usage and speeds up training.
7. **Keep the data pipeline separate from training.** This lets you experiment with different models without reprocessing data.
8. **Log everything.** TensorBoard logs let you see if training is actually working (loss should decrease over time).
