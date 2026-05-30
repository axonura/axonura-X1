import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dotenv import load_dotenv
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import wandb
from inference import Model, Pipeline, utils

VOCAB_SIZE = 50000
DIM = 10240
HEADS = 20480
LAYERS = 40960
MAX_LEN = 1024
DROPOUT = 0.1
BATCH_SIZE = 512
EPOCHS = 100
DEPTH_RATE = 64
TOKENIZER_PATH = "tokenizer.json"

if os.path.exists("model.weights.pt"):
    os.remove("model.weights.pt")
if os.path.exists("tokenizer.json"):
    os.remove("tokenizer.json")

if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "axonura-x1")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if WANDB_API_KEY is None:
    user_input = input("Enter API Key To Visualize (Optional): ").strip()
    WANDB_API_KEY = user_input if user_input else None


class ThinkingGPTModule(pl.LightningModule):
    def __init__(self, vocab_size, depthRate, dim, heads, layers, dropout, max_len):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model.ThinkingGPT(
            vocab_size=vocab_size,
            depthRate=depthRate,
            dim=dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, kv_cache=None):
        return self.model(input_ids, kv_cache=kv_cache)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        acc = (logits.argmax(-1) == y).float().mean()
        learning_error = 1.0 - acc
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/learning_error", learning_error, on_step=True, on_epoch=True)
        self.log("train/learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        acc = (logits.argmax(-1) == y).float().mean()
        learning_error = 1.0 - acc
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("val/learning_error", learning_error, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

print("Building tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
)

tokenizer.train_from_iterator(utils.batch_iterator(dataset["train"]), trainer=trainer)
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.pad_token = "<pad>"

print("Preparing data pipeline...")
pipeline = Pipeline.DSPipeline(dataset, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
train_loader, val_loader = pipeline.call()

train_samples = len(dataset["train"])
val_samples = len(dataset["validation"]) if "validation" in dataset else 0
STEPS_PER_EPOCH = min(100, train_samples // BATCH_SIZE)
VALIDATION_STEPS = min(20, val_samples // BATCH_SIZE) if val_samples > 0 else None

print(f"Building ThinkingGPT module (Steps: {STEPS_PER_EPOCH}, Val Steps: {VALIDATION_STEPS})...")
lightning_module = ThinkingGPTModule(
    vocab_size=VOCAB_SIZE,
    depthRate=DEPTH_RATE,
    dim=DIM,
    heads=HEADS,
    layers=LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN,
)

wandb_logger = None
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    if not WANDB_ENTITY:
        try:
            api = wandb.Api()
            WANDB_ENTITY = api.viewer.get("username") or api.default_entity
            print(f"Using WandB entity: {WANDB_ENTITY}")
        except Exception as e:
            print(f"Warning: Could not fetch default entity from WandB API: {e}")
            WANDB_ENTITY = None

    wandb_logger = WandbLogger(
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
    wandb_logger.watch(lightning_module, log="all", log_freq=100)
else:
    print("WANDB_API_KEY not found in environment. Skipping Weights & Biases logging.")

callbacks = []
if wandb_logger is not None:
    callbacks.append(ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1))

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    logger=wandb_logger,
    callbacks=callbacks,
    limit_train_batches=STEPS_PER_EPOCH,
    limit_val_batches=VALIDATION_STEPS,
    log_every_n_steps=1,
    accelerator="auto",
    devices="auto",
)

print("Starting training...")
trainer.fit(lightning_module, train_loader, val_loader)

print("Saving model...")
torch.save(lightning_module.model.state_dict(), "model.weights.pt")
if wandb_logger is not None:
    wandb.finish()
print("AI building logic completed successfully.")
