import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_wavelength=10000):
        super().__init__()
        inv_freq = 1.0 / (max_wavelength ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return x * cos + self._rotate_half(x) * sin

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, depth, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = max(1, dim // heads)
        self.scale = self.head_dim ** -0.5
        self.depth = depth
        self.dropout_rate = dropout
        self.use_qk_norm = False
        self.use_sliding_window = False
        self.window_size = 64

        self.rope = RotaryEmbedding(self.head_dim, max_wavelength=10000)
        self.query_dense = nn.Linear(dim, dim, bias=False)
        self.key_dense = nn.Linear(dim, dim, bias=False)
        self.value_dense = nn.Linear(dim, dim, bias=False)
        self.output_dense = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, x, kv_cache=None, mask=None):
        B, T, C = x.shape

        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)

        if self.use_qk_norm:
            q = self.q_norm(q.view(B, T, self.heads, self.head_dim)).view(B, T, C)
            k = self.k_norm(k.view(B, T, self.heads, self.head_dim)).view(B, T, C)

        q = q.view(B, T, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim)
        v = v.view(B, T, self.heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=1)
            v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        S = attn.size(-1)
        causal_mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        attn = attn.masked_fill(causal_mask == 0, -1e9)

        attn = F.normalize(attn, p=2, dim=-1) * self.depth
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output_dense(out)

        return out, kv_cache


class FeedForward(nn.Module):
    def __init__(self, d_model=256, multiplier=2.66, dropout=0.1):
        super().__init__()
        d_ff = int(multiplier * d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.w1(x) * torch.sigmoid(self.w2(x))
        hidden = self.dropout(hidden)
        return self.w_out(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, depthRate, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)

        attn_depth_arg = (math.sin(dim * heads) + math.cos(dim * heads)) - dropout
        attn_depth = (math.log(max(attn_depth_arg, 1e-10)) / 8) * depthRate
        self.attn = CausalSelfAttention(
            dim=dim, heads=heads, dropout=dropout, depth=attn_depth
        )
        self.ffn = FeedForward(d_model=dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, mask=None):
        normed_x = self.norm1(x)
        attn_out, kv_cache = self.attn(normed_x, kv_cache=kv_cache, mask=mask)
        x = x + self.dropout(attn_out)

        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)
        return x, kv_cache


class ThinkingGPT(nn.Module):
    def __init__(self, vocab_size, depthRate, dim=256, heads=8, layers=4, dropout=0.1, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(depthRate=depthRate, dim=dim, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.norm_final = nn.LayerNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None, mask=None):
        x = self.embedding(input_ids)

        for block in self.blocks:
            x, kv_cache = block(x, kv_cache=kv_cache, mask=mask)

        x = self.norm_final(x)
        logits = self.head(x)
        return logits
