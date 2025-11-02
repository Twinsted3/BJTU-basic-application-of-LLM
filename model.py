import math
import torch
import torch.nn as nn


class MyTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        B, T, D = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # 分qkv到多头
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(context)
        return out


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, use_residual=True, use_layernorm=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout)

        if self.use_layernorm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # self-attention
        attn_out = self.attn(x, mask=mask)
        if self.use_residual:
            x = x + attn_out
        else:
            x = attn_out
        if self.use_layernorm:
            x = self.ln1(x)

        # FFN
        out = self.ffn(x)
        if self.use_residual:
            x = x + out
        else:
            x = out
        if self.use_layernorm:
            x = self.ln2(x)
        return x


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=4, n_head=4, d_ff=1024, max_len=512, dropout=0.1, use_positional_encoding=True, use_residual=True, use_layernorm=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = (PositionalEncoding(d_model, max_len=max_len) if use_positional_encoding else None)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff, dropout=dropout, use_residual=use_residual, use_layernorm=use_layernorm) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.size()
        x = self.token_emb(idx) * math.sqrt(self.d_model)  # (B, T, D)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # mask shape (B, 1, T, T) or (1,1,T,T)
        device = x.device
        causal_mask = (torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0))  # (1,1,T,T)
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    # 生成一个上三角矩阵的注意力掩码，防止模型看到未来的token
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(device)
