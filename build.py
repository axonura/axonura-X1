import os
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import wandb
from wandb.keras import WandbCallback
from inference import Model, Pipeline, utils

# Configuration
VOCAB_SIZE = 10000
DIM = 256
HEADS = 8
LAYERS = 4
MAX_LEN = 128
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 25
TOKENIZER_PATH = "tokenizer.json"

load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "axonura-x1-training")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

wandb_run = None
wandb_callback = None

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "vocab_size": VOCAB_SIZE,
            "dim": DIM,
            "heads": HEADS,
            "layers": LAYERS,
            "max_len": MAX_LEN,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "tokenizer_path": TOKENIZER_PATH,
        },
    )
    wandb_callback = WandbCallback(log_weights=True, log_gradients=True)
else:
    print("WANDB_API_KEY not found in environment. Skipping Weights & Biases logging.")

class WandbMetricsCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if wandb_run is None:
            return
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        lr_value = lr.numpy() if hasattr(lr, "numpy") else tf.keras.backend.get_value(lr)
        accuracy = logs.get("accuracy")
        learning_error = 1.0 - float(accuracy) if accuracy is not None else None
        wandb.log(
            {
                "train/loss": logs.get("loss"),
                "train/accuracy": accuracy,
                "train/learning_error": learning_error,
                "train/learning_rate": lr_value,
            },
            commit=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        if wandb_run is None:
            return
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy")
        val_learning_error = 1.0 - float(val_accuracy) if val_accuracy is not None else None
        wandb.log(
            {
                "val/loss": logs.get("val_loss"),
                "val/accuracy": val_accuracy,
                "val/learning_error": val_learning_error,
            }
        )

# Load Dataset
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

print("Building tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
)
            
tokenizer.train_from_iterator(
    utils.batch_iterator(dataset["train"]),
    trainer=trainer
)
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

# Load the fast tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.pad_token = "<pad>"

# Data Pipeline
print("Preparing data pipeline...")
pipeline = Pipeline.DSPipeline(dataset, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
train_ds, val_ds = pipeline.call()

# Calculate Dynamic Steps
train_samples = len(dataset["train"])
val_samples = len(dataset["validation"]) if "validation" in dataset else 0

# Limit steps for a fast 'building' phase or set to full dataset
STEPS_PER_EPOCH = min(100, train_samples // BATCH_SIZE)
VALIDATION_STEPS = min(20, val_samples // BATCH_SIZE) if val_samples > 0 else None

# 5. Build Model
print(f"Building ThinkingGPT model (Steps: {STEPS_PER_EPOCH}, Val Steps: {VALIDATION_STEPS})...")
model = Model.ThinkingGPT(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    heads=HEADS,
    layers=LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Enable Wandb gradient/parameter tracking across all model layers.
if wandb_run is not None:
    wandb.watch(model, log="all", log_freq=100)

# 6. Training
print("Starting training...")
callbacks = [WandbMetricsCallback()]
if wandb_callback is not None:
    callbacks.append(wandb_callback)
model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 7. Save Model
print("Saving model...")
model.save_weights("model.weights.h5")
if wandb_run is not None:
    wandb_run.finish()
print("AI building logic completed successfully.")
