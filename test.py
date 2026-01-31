import os
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

def predict(prompt, temprature=0.7, max_len=128, max_tokens=512):
    # Prepend BOS token to the prompt
    prompt_with_bos = "<bos>" + prompt
    
    enc = tokenizer(
        [prompt_with_bos],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )

    ids = enc["input_ids"]

    for _ in range(max_tokens):
        logits = model(ids, training=False)
        logits = logits[:, -1, :] / temprature

        # Top-K Sampling
        values, top_k_indices = tf.math.top_k(logits, k=64)
        probs = tf.math.softmax(values, axis=-1)

        # Sample from the top-k indices
        sample_index = tf.random.categorical(probs, num_samples=1)
        NXID = tf.gather(top_k_indices, sample_index, batch_dims=1)

        ids = tf.concat([ids, NXID], axis=-1)

        # End The Prediction If Next Token is EOS
        if NXID[0, 0] == tokenizer.eos_token_id:
            break

    # Decode and skip special tokens (removes <bos>, <eos>, <pad>, etc.)
    return tokenizer.decode(ids[0].numpy(), skip_special_tokens=True)

while True:
    prompt = input("You: ")
    if(prompt.lower() == "exit"):
        os.exit()
    elif(prompt.lower() == "clear"):
        os.system("clear || cls")
    else:
        print("AI: ", predict(prompt.lower()))