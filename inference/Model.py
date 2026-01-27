import datasets
import keras_hub
import tensorflow as tf
from tensorflow import keras

class CausalSelfAttention(keras.layers.Layer):
    def __init__(self, dim=256, heads=8, dropout=0.1, rope_dim=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Layers
        self.qkv = keras.layers.Dense(dim * 3, use_bias=False)
        self.proj = keras.layers.Dense(dim)
        self.dropout = keras.layers.Dropout(dropout)

        # Rotary embedding (applied to Q and K)
        self.rope = keras_hub.layers.RotaryEmbedding(
            max_wavelength=10000
        )

    def call(self, x, kv_cache=None, training=False):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)

        BATCH = tf.shape(x)[0]
        SEQL = tf.shape(x)[1]

        # QKV projection
        qkv = self.qkv(x)  # (B, S, 3*dim)
        qkv = tf.reshape(qkv, (BATCH, SEQL, 3, self.heads, self.head_dim))
        Q, K, V = tf.unstack(qkv, axis=2)  # each: (B, S, H, D)

        # Apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # Handle KV cache for autoregressive decoding (Crucial For inference/answering)
        if kv_cache is not None:
            # Concatenate new keys/values to cache
            K = tf.concat([kv_cache["k"], K], axis=1)
            V = tf.concat([kv_cache["v"], V], axis=1)
            kv_cache["k"], kv_cache["v"] = K, V

        # Transpose for attention
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, H, S, D)
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        # Attention scores
        attn = tf.matmul(Q, K, transpose_b=True) * self.scale  # (B, H, S, S)

        # Causal mask
        seq_len = tf.shape(attn)[-1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
        attn = tf.where(mask == 0, -1e9, attn)

        # Softmax And Dropout
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        # Weighted sum
        out = tf.matmul(attn, V)

        out = tf.transpose(out, [0, 2, 1, 3])  # (B, S, H, D)
        out = tf.reshape(out, (BATCH, SEQL, self.dim))

        return self.proj(out), kv_cache

class FeedForward(keras.layers.Layer):
    def __init__(self, d_model=256, multiplier=2.66, dropout=0.1):
        super().__init__()
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
    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.attn = CausalSelfAttention(dim=dim, heads=heads, dropout=dropout)
        self.ffn = FeedForward(d_model=dim, dropout=dropout)

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x, kv_cache=None, training=False):
        normed_x = self.norm1(x)
        attn_out, kv_cache = self.attn(normed_x, kv_cache=kv_cache, training=training)
        x = x + self.dropout(attn_out, training=training)  # residual connection

        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x, training=training)
        x = x + self.dropout(ffn_out, training=training)  # residual connection

        return x, kv_cache

class ThinkingGPT(keras.Model):
    def __init__(self, vocab_size, dim=256, heads=8, layers=4, dropout=0.1, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.embedding = keras.layers.Embedding(vocab_size, dim)
        self.blocks = [TransformerBlock(dim=dim, heads=heads, dropout=dropout) for _ in range(layers)]
        self.norm_final = keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, x, kv_cache=None, training=False):
        x = self.embedding(x)
        
        for block in self.blocks:
            x, kv_cache = block(x, kv_cache=kv_cache, training=training)
            
        x = self.norm_final(x)
        logits = self.head(x)
        
        return logits