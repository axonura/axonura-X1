import os, sys, re
import torch
from transformers import PreTrainedTokenizerFast
from inference import Model, utils

VOCAB_SIZE = 50000
DIM = 10240
HEADS = 20480
LAYERS = 40960
MAX_LEN = 1024
DROPOUT = 0.1
DEPTH_RATE = 64

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
model.load_state_dict(torch.load("model.weights.pt", map_location=device))
model.eval()

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"


@torch.no_grad()
def predict(prompt, temperature=0.7, max_tokens=256):
    prompt_with_bos = tokenizer.bos_token + prompt
    enc = tokenizer(prompt_with_bos, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)

    for _ in range(max_tokens):
        logits = model(ids)
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
