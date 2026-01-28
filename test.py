import tensorflow as tf
import tensorflow.keras as keras
from transformers import PreTrainedTokenizerFast

# Loads The Model And Tokenizer
model = tf.keras.models.load_model("model.h5")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

def predict(prompt, temprature=0.7, max_len=128, max_tokens=2048):
    enc = tokenizer(
        [prompt],
        padding="<pad>",
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
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if NXID[0, 0] == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0].numpy())

while True:
    prompt = input("You: ")
    if(prompt.lower() == "exit"):
        break
    print("AI: ", predict(prompt.lower()))