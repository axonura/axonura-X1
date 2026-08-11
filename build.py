# Copyright 2026 First Person
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dotenv import load_dotenv
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import wandb
from inference import Model, Pipeline, utils, datasets

VOCAB_SIZE = 50000
DIM = 256
HEADS = 8
LAYERS = 4
MAX_LEN = 512
DROPOUT = 0.1
BATCH_SIZE = 8
EPOCHS = 100
DEPTH_RATE = 64
TOKENIZER_PATH = "tokenizer.json"
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<image>", "<audio>", "<video>"]

parser = argparse.ArgumentParser(description="Build Axonura X1")
parser.add_argument(
    "--modality",
    default="text",
    choices=["text", "images", "videos", "audio", "documents",
             "spreadsheets", "presentations", "markdown", "local"],
    help="Which dataset modality to train on.",
)
parser.add_argument(
    "--local-folder",
    default=None,
    help="Local folder to ingest when --modality local is used.",
)
args = parser.parse_args()

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
    def __init__(
        self,
        vocab_size,
        depthRate,
        dim,
        heads,
        layers,
        dropout,
        max_len,
        image_token_id=None,
        audio_token_id=None,
        video_token_id=None,
    ):
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
            image_token_id=image_token_id,
            audio_token_id=audio_token_id,
            video_token_id=video_token_id,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id

    def forward(self, batch, kv_cache=None):
        if isinstance(batch, dict):
            return self.model(
                batch["input_ids"],
                vision_features=batch.get("vision_features"),
                audio_features=batch.get("audio_features"),
                vision_mask=batch.get("vision_mask"),
                audio_mask=batch.get("audio_mask"),
                kv_cache=kv_cache,
            )
        return self.model(batch, kv_cache=kv_cache)

    def _masked_loss(self, logits, batch):
        logits = logits.reshape(-1, logits.size(-1))
        labels = batch["labels"].reshape(-1)
        loss_mask = batch["loss_mask"].reshape(-1).float()
        per_token = self.loss_fn(logits, labels)
        denom = loss_mask.sum().clamp(min=1.0)
        loss = (per_token * loss_mask).sum() / denom
        acc = ((logits.argmax(-1) == labels) & loss_mask.bool()).float().sum() / denom
        return loss, acc

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            logits = self.model(
                batch["input_ids"],
                vision_features=batch.get("vision_features"),
                audio_features=batch.get("audio_features"),
                vision_mask=batch.get("vision_mask"),
                audio_mask=batch.get("audio_mask"),
            )
            loss, acc = self._masked_loss(logits, batch)
        else:
            x, y = batch
            logits = self.model(x)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).mean()
            acc = (logits.argmax(-1) == y).float().mean()
        learning_error = 1.0 - acc
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/learning_error", learning_error, on_step=True, on_epoch=True)
        self.log("train/learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            logits = self.model(
                batch["input_ids"],
                vision_features=batch.get("vision_features"),
                audio_features=batch.get("audio_features"),
                vision_mask=batch.get("vision_mask"),
                audio_mask=batch.get("audio_mask"),
            )
            loss, acc = self._masked_loss(logits, batch)
        else:
            x, y = batch
            logits = self.model(x)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).mean()
            acc = (logits.argmax(-1) == y).float().mean()
        learning_error = 1.0 - acc
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("val/learning_error", learning_error, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


print("Loading dataset...")
if args.modality == "local":
    if not args.local_folder:
        raise SystemExit("--local-folder is required when --modality local is used.")
    print(f"Ingesting local folder: {args.local_folder}")
    shards = list(datasets.ingest_folder(args.local_folder, includeMedia=False))
    dataset = datasets.rows_to_dataset(shards).train_test_split(test_size=0.05)
elif args.modality == "text":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
else:
    print(f"Downloading {args.modality} dataset from approved sources...")
    dataset = datasets.load_modality(args.modality)

print("Building tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
)

if args.modality == "text":
    tokenizer.train_from_iterator(utils.batch_iterator(dataset["train"]), trainer=trainer)
else:
    def text_iterator(split):
        for row in split:
            text = row.get("text")
            if text and text.strip():
                yield text

    tokenizer.train_from_iterator(text_iterator(dataset["train"]), trainer=trainer)
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")
AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<audio>")
VIDEO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<video>")
MEDIA_TOKEN_IDS = (IMAGE_TOKEN_ID, AUDIO_TOKEN_ID, VIDEO_TOKEN_ID)

print("Preparing data pipeline...")
pipeline = Pipeline.DSPipeline(
    dataset,
    tokenizer,
    max_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    media_token_ids=MEDIA_TOKEN_IDS,
)
train_loader, val_loader = pipeline.call()

train_samples = len(dataset["train"])
val_samples = len(dataset["validation"]) if "validation" in dataset else len(dataset["test"]) if "test" in dataset else 0
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
    image_token_id=IMAGE_TOKEN_ID,
    audio_token_id=AUDIO_TOKEN_ID,
    video_token_id=VIDEO_TOKEN_ID,
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
