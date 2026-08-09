# Copyright 2026 First Person
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_wavelength=10000):
        super().__init__()
        half = max(1, dim // 2)
        inv_freq = 1.0 / (max_wavelength ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq[:half], persistent=False)

    def forward(self, x):
        head_dim = x.shape[-1]
        half = head_dim // 2
        if half == 0:
            return x
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq[:half])
        cos, sin = freqs.cos(), freqs.sin()
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


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
        T = attn.size(-2)
        causal_mask = torch.tril(torch.ones(T, S, device=x.device)).view(1, 1, T, S)
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


class VisionAdapter(nn.Module):
    def __init__(self, encoderDim=768, dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(encoderDim, eps=1e-5)
        self.proj = nn.Linear(encoderDim, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, mask=None):
        x = self.norm(features)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            x = x.sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        return self.dropout(self.act(self.proj(x)))


class AudioAdapter(nn.Module):
    def __init__(self, melBins=80, timeSteps=3000, dim=256, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(melBins, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()
        self.proj = nn.Linear(256, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, mask=None):
        batchSize, chunks, melBins, timeSteps = features.shape
        x = features.view(batchSize * chunks, melBins, timeSteps)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.mean(dim=2).view(batchSize, chunks, -1)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            x = x.sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        return self.dropout(self.norm(self.proj(x)))


class ThinkingGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        depthRate,
        dim=256,
        heads=8,
        layers=4,
        dropout=0.1,
        max_len=128,
        encoder_dim=768,
        image_token_id=None,
        audio_token_id=None,
        video_token_id=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(depthRate=depthRate, dim=dim, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.norm_final = nn.LayerNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.vision_adapter = VisionAdapter(encoderDim=encoder_dim, dim=dim, dropout=dropout)
        self.audio_adapter = AudioAdapter(dim=dim, dropout=dropout)

    def _splice_media(self, x, input_ids, mediaId, features, mask, adapter):
        if mediaId is None or features is None:
            return x
        positions = input_ids == mediaId
        if not positions.any():
            return x
        adapterOut = adapter(features, mask)
        hasMedia = positions.any(dim=-1)
        firstIdx = positions.float().argmax(dim=-1)
        rows = torch.arange(x.shape[0], device=x.device)[hasMedia]
        x[rows, firstIdx[hasMedia], :] = adapterOut[rows]
        return x

    def forward(
        self,
        input_ids,
        vision_features=None,
        audio_features=None,
        vision_mask=None,
        audio_mask=None,
        kv_cache=None,
        mask=None,
    ):
        x = self.embedding(input_ids)

        x = self._splice_media(
            x, input_ids, self.image_token_id, vision_features, vision_mask, self.vision_adapter
        )
        x = self._splice_media(
            x, input_ids, self.video_token_id, vision_features, vision_mask, self.vision_adapter
        )
        x = self._splice_media(
            x, input_ids, self.audio_token_id, audio_features, audio_mask, self.audio_adapter
        )

        for block in self.blocks:
            x, kv_cache = block(x, kv_cache=kv_cache, mask=mask)

        x = self.norm_final(x)
        logits = self.head(x)
        return logits
