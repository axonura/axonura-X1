import tensorflow as tf
from functools import partial

def text_gen(dataset_shard):
    """Generator that yields text from a specific dataset shard/split"""
    for x in dataset_shard:
        text = x.get("text", "")
        if text and text.strip():
            yield text

def encode_batch(text_batch, tokenizer, max_len):
    """Encodes a batch of text using the tokenizer"""
    # Decode texts from tensors if they come from TF dataset
    texts = [t.numpy().decode("utf-8") for t in text_batch]

    # Filter out empty strings (replace with space to avoid errors)
    texts = [t if t.strip() else " " for t in texts]

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )
    return enc["input_ids"]

def tf_encode(text_batch, tokenizer, max_len):
    """TensorFlow wrapper for the encoding function"""
    # Create a partial function with tokenizer and max_len pinned
    encode_fn = partial(encode_batch, tokenizer=tokenizer, max_len=max_len)
    
    ids = tf.py_function(encode_fn, [text_batch], tf.int32)
    ids.set_shape([None, max_len]) 
    return ids

def batch_iterator(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]