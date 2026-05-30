import math
import datasets
import keras_hub
import tensorflow as tf
from tensorflow import keras

class CausalSelfAttention(keras.layers.Layer):
    def __init__(self, depth, dim=256, heads=8, dropout=0.1, rope_dim=None, name="causal_self_attention"):
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads
        self.head_dim = max(1, dim // heads)
        self.scale = self.head_dim ** -0.5
        self.depth = dim / self.scale * depth
        self.dropout_rate = dropout
        self.use_qk_norm = False
        self.use_sliding_window = False
        self.window_size = 64

    def build(self, input_shape):
        super().build(input_shape)
        self.rope = keras_hub.layers.RotaryEmbedding(max_wavelength=10000)
        self.query_dense = keras.layers.Dense(self.dim, use_bias=False, name="query")
        self.key_dense = keras.layers.Dense(self.dim, use_bias=False, name="key")
        self.value_dense = keras.layers.Dense(self.dim, use_bias=False, name="value")
        self.output_dense = keras.layers.Dense(self.dim, use_bias=False, name="output")
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        if self.use_qk_norm:
            self.q_norm = keras.layers.LayerNormalization(epsilon=1e-6)
            self.k_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.built = True

    def call(self, inputs, kv_cache=None, training=False, return_attention_weights=False, mask=None):
        x = inputs
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)

        BATCH = tf.shape(x)[0]
        SEQL = tf.shape(x)[1]

        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = tf.reshape(q, (BATCH, SEQL, self.heads, self.head_dim))
        k = tf.reshape(k, (BATCH, SEQL, self.heads, self.head_dim))
        v = tf.reshape(v, (BATCH, SEQL, self.heads, self.head_dim))

        q = self.rope(q)
        k = self.rope(k)

        if kv_cache is not None:
            k = tf.concat([kv_cache["k"], k], axis=1)
            v = tf.concat([kv_cache["v"], v], axis=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        seq_len = tf.shape(attn)[-1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len))
        attn = tf.where(causal_mask == 0, -1e9, attn)

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (BATCH, SEQL, self.dim))

        return self.output_dense(out), kv_cache

class FeedForward(keras.layers.Layer):
    def __init__(self, d_model=256, multiplier=2.66, dropout=0.1, name="feed_forward"):
        super().__init__(name=name)
        # d_ff = SwiGLU multiplier × d_model
        d_ff = int(multiplier * d_model)

        # SwiGLU projections
        self.w1 = keras.layers.Dense(d_ff, use_bias=False)
        self.w2 = keras.layers.Dense(d_ff, use_bias=False)
        self.w_out = keras.layers.Dense(d_model, use_bias=False)

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # SwiGLU activation
        hidden = self.w1(x) * tf.nn.sigmoid(self.w2(x))
        hidden = self.dropout(hidden, training=training)
        return self.w_out(hidden)

class TransformerBlock(keras.layers.Layer):
    def __init__(self, depthRate,dim=256, heads=8, dropout=0.1, name="transformer_block"):
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        attn_depth_arg = (math.sin(dim * heads) + math.cos(dim * heads)) - dropout
        attn_depth = (math.log(max(attn_depth_arg, 1e-10)) / 8) * depthRate
        self.attn = CausalSelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            name=f"{name}_attention",
            depth=attn_depth
        )
        self.ffn = FeedForward(
            d_model=dim,
            dropout=dropout,
            name=f"{name}_ffn"
        )

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, mask=None, kv_cache=None, return_attention_weights=False):
        x = inputs
        training = bool(training) if training is not None else False
        normed_x = self.norm1(x)
        attn_result = self.attn(normed_x, training=training, 
                               return_attention_weights=return_attention_weights, 
                               kv_cache=kv_cache)
        
        if return_attention_weights:
            attn_out, kv_cache, attn_weights = attn_result
        else:
            attn_out, kv_cache = attn_result
            attn_weights = None
            
        x = x + self.dropout(attn_out, training=training)  # residual connection

        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x, training=training)
        x = x + self.dropout(ffn_out, training=training)  # residual connection

        if return_attention_weights:
            return x, kv_cache, attn_weights
        return x, kv_cache

class ThinkingGPT(keras.Model):
    def __init__(self, vocab_size, depthRate, dim=256, heads=8, layers=4, dropout=0.1, max_len=128, name="thinking_gpt"):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.embedding = keras.layers.Embedding(vocab_size, dim, name=f"{name}_embedding")
        self.blocks = [
            TransformerBlock(depthRate=depthRate, dim=dim, heads=heads, dropout=dropout, name=f"{name}_block_{idx}")
            for idx in range(layers)
        ]
        self.norm_final = keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, inputs, training=None, mask=None, kv_cache=None, return_attention_weights=False):
        """
        Forward pass of the model.
        
        Args:
            inputs: Input token IDs of shape (batch_size, seq_len)
            training: Whether in training mode
            mask: Optional mask (Keras API compatibility)
            kv_cache: Optional KV cache for autoregressive generation
            return_attention_weights: Whether to return attention weights for analysis
        
        Returns:
            Model logits or tuple of (logits, attention_weights) if return_attention_weights=True
        """
        x = inputs
        x = self.embedding(x)
        
        all_attn_weights = []
        for block in self.blocks:
            result = block(x, training=training, mask=mask,
                          kv_cache=kv_cache, 
                          return_attention_weights=return_attention_weights)
            if return_attention_weights:
                x, kv_cache, attn_weights = result
                all_attn_weights.append(attn_weights)
            else:
                x, kv_cache = result
            
        x = self.norm_final(x)
        logits = self.head(x)
        
        if return_attention_weights:
            return logits, all_attn_weights
        return logits
