import os, sys, re
import tensorflow as tf
from transformers import PreTrainedTokenizerFast
from inference import Model

# Configuration (must match build.py)
VOCAB_SIZE = 10000
DIM = 256
HEADS = 8
LAYERS = 4
MAX_LEN = 128
DROPOUT = 0.1

# Instantiate the Model
model = Model.ThinkingGPT(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    heads=HEADS,
    layers=LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
)

# Build the model by calling it on dummy input
dummy_input = tf.zeros((1, MAX_LEN), dtype=tf.int32)
_ = model(dummy_input, training=False)

# Load the weights
model.load_weights("model.weights.h5")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# Configure special tokens
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

def predict(prompt, temperature=0.7, max_tokens=256):
    # Add BOS token
    prompt_with_bos = tokenizer.bos_token + prompt

    # Encode without padding
    enc = tokenizer(
        prompt_with_bos,
        return_tensors="tf",
        add_special_tokens=False
    )
    ids = enc["input_ids"]

    for _ in range(max_tokens):
        logits = model(ids, training=False)
        logits = logits[:, -1, :] / temperature

        # Top-K sampling
        values, indices = tf.math.top_k(logits, k=64)
        probs = tf.nn.softmax(values)
        sample = tf.random.categorical(probs, 1)
        next_id = tf.gather(indices, sample, batch_dims=1)

        # Append the next token
        ids = tf.concat([ids, next_id], axis=-1)

        # Stop if EOS is generated
        if next_id[0, 0] == tokenizer.eos_token_id:
            break

    # Decode to text
    text = tokenizer.decode(ids[0].numpy(), skip_special_tokens=True)

    # Post-process GPT/BPE subwords:
    text = re.sub(r'(\w) (\w+)', r'\1\2', text)
    text = text.replace("Ġ", " ")

    # Strip leading/trailing whitespace
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
        print('Type Anything Then Press Enter Key To Ask AI')
    else:
        print("AI: ", predict(prompt))