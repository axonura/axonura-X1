import tensorflow as tf
from functools import partial
from . import utils

class DSPipeline:
    def __init__(self, dataset, tokenizer, max_len=128, batch_size=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        # Assuming dataset has train/test splits or similar structure
        self.train_data = dataset.get("train", dataset) 
        self.test_data = dataset.get("test", None)

    def call(self):
        # -----------------------------
        # Build TensorFlow dataset for training
        # -----------------------------
        
        # Create a partial for the generator to pass the specific dataset split
        train_gen = partial(utils.text_gen, dataset_shard=self.train_data)
        
        train_ds = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
        )

        train_ds = train_ds.batch(self.batch_size, drop_remainder=True)
        
        # Create a partial for the encoder to pass tokenizer and max_len
        encode_fn = partial(utils.tf_encode, tokenizer=self.tokenizer, max_len=self.max_len)
        
        train_ds = train_ds.map(encode_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(
            lambda x: (x[:, :-1], x[:, 1:]),  # input-target shift
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if self.test_data:
            # -----------------------------
            # Build TensorFlow dataset for validation
            # -----------------------------
            val_gen = partial(utils.text_gen, dataset_shard=self.test_data)
            
            val_ds = tf.data.Dataset.from_generator(
                val_gen,
                output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
            )

            val_ds = val_ds.batch(self.batch_size, drop_remainder=True)
            val_ds = val_ds.map(encode_fn, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(
                lambda x: (x[:, :-1], x[:, 1:]),  # input-target shift
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds