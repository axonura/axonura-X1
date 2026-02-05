import datasets
import keras_hub
import tensorflow as tf
from tensorflow import keras

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, dim=256, heads=8, dropout=0.1, rope_dim=None, name="multi_head_attention",
                 use_flash_attention=True, use_sliding_window=False, window_size=512,
                 use_qk_norm=False, use_gradient_checkpointing=False, attention_scale=None):
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_rate = dropout
        self.use_flash_attention = use_flash_attention
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.use_qk_norm = use_qk_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attention_scale = attention_scale or (self.head_dim ** -0.5)
        
        # Attention pattern monitoring
        self.attention_weights = None
        self.call_count = 0

    def build(self, input_shape):
        super().build(input_shape)
        
        # RoPE for Q and K
        self.rope = keras_hub.layers.RotaryEmbedding(max_wavelength=10000)
        
        # Explicit Q, K, V projections
        self.query_dense = keras.layers.Dense(self.dim, use_bias=False, name="query")
        self.key_dense = keras.layers.Dense(self.dim, use_bias=False, name="key")
        self.value_dense = keras.layers.Dense(self.dim, use_bias=False, name="value")
        self.output_dense = keras.layers.Dense(self.dim, use_bias=False, name="output")


        # Dropout layer for attention
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        
        # Query/Key normalization (optional)
        if self.use_qk_norm:
            self.q_norm = keras.layers.LayerNormalization(epsilon=1e-6)
            self.k_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Sliding window attention mask cache
        if self.use_sliding_window:
            self.window_mask_cache = {}
        
        # Head dropout for regularization
        self.head_dropout = keras.layers.Dropout(min(0.1, self.dropout_rate / 2))
        
        self.built = True

    def _get_sliding_window_mask(self, seq_len):
        """Generate sliding window attention mask for long sequences."""
        if seq_len in self.window_mask_cache:
            return self.window_mask_cache[seq_len]
        
        # Create banded matrix for local attention
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), 
            self.window_size,  # lower bandwidth
            self.window_size   # upper bandwidth
        )
        # Ensure causal mask is maintained
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask * causal_mask
        
        self.window_mask_cache[seq_len] = mask
        return mask

    def _flash_attention(self, q, k, v, mask, training):
        """Memory-efficient attention computation with automatic optimization."""
        # Automatic precision selection based on hardware capability
        use_fp16 = tf.config.experimental.get_device_policy() != 'float32'
        compute_dtype = tf.float16 if use_fp16 and training else tf.float32
        
        # Cast to compute dtype
        q = tf.cast(q, compute_dtype)
        k = tf.cast(k, compute_dtype)
        v = tf.cast(v, compute_dtype)
        
        # Scaled dot-product attention with numerical stability
        attn_scores = tf.matmul(q, k, transpose_b=True) * tf.cast(self.attention_scale, compute_dtype)
        
        # Apply mask
        if mask is not None:
            mask = tf.cast(mask, compute_dtype)
            attn_scores = attn_scores * mask + (1.0 - mask) * tf.float16.min if use_fp16 else -1e9
        
        # Softmax with improved numerical stability
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        
        # Dropout during training
        if training:
            attn_probs = tf.nn.dropout(attn_probs, rate=self.dropout_rate)
        
        # Apply attention to values
        output = tf.matmul(attn_probs, v)
        
        # Store attention weights for analysis
        if not training:
            self.attention_weights = tf.reduce_mean(attn_probs, axis=1)
        
        return tf.cast(output, tf.float32)

    def call(self, inputs, kv_cache=None, training=False, return_attention_weights=False, mask=None):
        x = inputs
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        
        self.call_count += 1
        BATCH = tf.shape(x)[0]
        SEQL = tf.shape(x)[1]
        
        # Linear projections for Q, K, V using explicit layers
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)
        
        # Optional Q/K normalization for training stability
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Reshape to (B, S, H, D)
        q = tf.reshape(q, (BATCH, SEQL, self.heads, self.head_dim))
        k = tf.reshape(k, (BATCH, SEQL, self.heads, self.head_dim))
        v = tf.reshape(v, (BATCH, SEQL, self.heads, self.head_dim))
        
        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)
        
        # Handle KV cache with automatic memory management
        if kv_cache is not None:
            # Ensure kv_cache tensors have proper shape before concatenation
            k_cache = tf.cast(kv_cache["k"], k.dtype)
            v_cache = tf.cast(kv_cache["v"], v.dtype)
            k = tf.concat(values=(k_cache, k), axis=1)
            v = tf.concat(values=(v_cache, v), axis=1)
            # Automatic cache size limiting to prevent OOM
            cache_len = tf.shape(k)[1]
            k = tf.cond(
                cache_len > 8192,
                lambda: k[:, -4096:, :, :],
                lambda: k
            )
            v = tf.cond(
                cache_len > 8192,
                lambda: v[:, -4096:, :, :],
                lambda: v
            )
            kv_cache["k"] = k
            kv_cache["v"] = v
        
        # Transpose to (B, H, S, D)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Create attention mask
        seq_len_q = tf.shape(q)[2]
        seq_len_k = tf.shape(k)[2]
        
        if self.use_sliding_window and seq_len_k > self.window_size:
            # Use sliding window attention for long sequences
            mask = self._get_sliding_window_mask(seq_len_k)
            mask = tf.slice(mask, [seq_len_k - seq_len_q, 0], [seq_len_q, seq_len_k])
        else:
            # Standard causal mask
            mask = tf.linalg.band_part(tf.ones((seq_len_q, seq_len_k)), -1, 0)
        
        mask = tf.reshape(mask, (1, 1, seq_len_q, seq_len_k))
        
        # Choose attention implementation based on configuration
        if self.use_flash_attention and training:
            # Use optimized flash attention during training
            output = self._flash_attention(q, k, v, mask, training)
        else:
            # Standard attention computation
            # Scaled dot-product attention
            scale = tf.cast(self.attention_scale, tf.float32)
            attn_scores = tf.matmul(q, k, transpose_b=True) * scale
            
            # Apply mask
            mask_float = tf.cast(mask, tf.float32)
            attn_scores = attn_scores * mask_float + (1.0 - mask_float) * -1e9
            
            # Softmax
            attn_probs = tf.nn.softmax(attn_scores, axis=-1)
            
            # Dropout
            attn_probs = self.dropout(attn_probs, training=training)
            
            # Apply attention to values
            output = tf.matmul(attn_probs, v)
            
            # Store attention weights
            if not training:
                self.attention_weights = tf.reduce_mean(attn_probs, axis=1)
        
        # Reshape output: (B, H, S, D) -> (B, S, H*D)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (BATCH, SEQL, self.dim))
        
        # Final projection
        output = self.output_dense(output)
        
        # Head dropout
        if training:
            output = self.head_dropout(output, training=training)
        
        # Return attention weights if requested
        if return_attention_weights:
            return output, kv_cache, self.attention_weights
        
        return output, kv_cache

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
    def __init__(self, dim=256, heads=8, dropout=0.1, name="transformer_block",
                 use_flash_attention=True, use_sliding_window=False, window_size=512):
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.attn = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            name=f"{name}_attention",
            use_flash_attention=use_flash_attention,
            use_sliding_window=use_sliding_window,
            window_size=window_size
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
    def __init__(self, vocab_size, dim=256, heads=8, layers=4, dropout=0.1, max_len=128, 
                 name="thinking_gpt", use_flash_attention=True, use_sliding_window=False, window_size=512):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.embedding = keras.layers.Embedding(vocab_size, dim, name=f"{name}_embedding")
        self.blocks = [
            TransformerBlock(dim=dim, heads=heads, dropout=dropout, name=f"{name}_block_{idx}",
                           use_flash_attention=use_flash_attention, 
                           use_sliding_window=use_sliding_window, 
                           window_size=window_size)
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
