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

import os, sys, re, argparse
import torch
from transformers import PreTrainedTokenizerFast
from inference import Model, utils

VOCAB_SIZE = 50000
DIM = 256
HEADS = 8
LAYERS = 4
MAX_LEN = 512
DROPOUT = 0.1
DEPTH_RATE = 64

parser = argparse.ArgumentParser(description="Chat with Axonura X1")
parser.add_argument(
    "--file", default=None,
    help="Optional media file (image/video/audio) to condition the model on.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model.ThinkingGPT(
    vocab_size=VOCAB_SIZE,
    depthRate=DEPTH_RATE,
    dim=DIM,
    heads=HEADS,
    layers=LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN,
)
model.to(device)
model.load_state_dict(torch.load("model.weights.pt", map_location=device), strict=False)
model.eval()

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids("<image>")
AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<audio>")
VIDEO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<video>")
model.image_token_id = IMAGE_TOKEN_ID
model.audio_token_id = AUDIO_TOKEN_ID
model.video_token_id = VIDEO_TOKEN_ID


def load_media_features(path):
    kind, *payload = utils.tokenizeFile(path)
    media_token = {"image": "<image>", "video": "<video>", "audio": "<audio>"}[kind]
    features = {}
    if kind in ("image", "video"):
        vis = payload[0].unsqueeze(0).to(device)
        features["vision_features"] = vis
        features["vision_mask"] = torch.ones(
            vis.shape[0], vis.shape[1], dtype=torch.bool, device=device
        )
    else:
        features["vision_features"] = None
        features["vision_mask"] = None

    audio_payload = payload[1] if kind == "video" else (payload[0] if kind == "audio" else None)
    if audio_payload is not None:
        aud = audio_payload.unsqueeze(0).to(device)
        features["audio_features"] = aud
        features["audio_mask"] = torch.ones(
            aud.shape[0], aud.shape[1], dtype=torch.bool, device=device
        )
    else:
        features["audio_features"] = None
        features["audio_mask"] = None
    return media_token, features


media_prefix = ""
media_features = {}
if args.file:
    media_prefix, media_features = load_media_features(args.file)
    print(f"Loaded media: {args.file} ({media_prefix})")


@torch.no_grad()
def predict(prompt, temperature=0.7, max_tokens=256):
    prompt_with_bos = tokenizer.bos_token + media_prefix + prompt
    enc = tokenizer(prompt_with_bos, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)

    for _ in range(max_tokens):
        logits = model(
            ids,
            vision_features=media_features.get("vision_features"),
            audio_features=media_features.get("audio_features"),
            vision_mask=media_features.get("vision_mask"),
            audio_mask=media_features.get("audio_mask"),
        )
        logits = logits[:, -1, :] / temperature

        next_id = utils.top_k_sampling(logits, k=64)
        ids = torch.cat([ids, next_id], dim=-1)

        if next_id[0, 0] == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(ids[0].cpu().numpy(), skip_special_tokens=True)
    text = re.sub(r"(\w) (\w+)", r"\1\2", text)
    text = text.replace("Ġ", " ")
    return text.strip()


print("Type Help To Get Commands")
while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        sys.exit()
    elif prompt.lower() == "clear":
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    elif prompt.lower() == "help":
        print("Type 'exit' with enter key to exit the program.")
        print("Type 'clear' with enter key to clear the screen.")
        print("Type Anything Then Press Enter Key To Ask AI")
    else:
        print("AI: ", predict(prompt))
