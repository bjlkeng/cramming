import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # As per Cramming, disable all QKV biases
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        queries = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        values = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)
        
        # Concatenate and linear transformation
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention layer
        residual = x
        x = self.self_attn(x, mask)
        x = self.dropout1(x)
        x += residual
        x = self.self_attn_layer_norm(x)
        
        # Feedforward layer
        residual = x
        x = self.ff(x)
        x = self.dropout2(x)
        x += residual
        x = self.ff_layer_norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    ''' Taken from the Pytorch tutorial:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModule(nn.Module):
    def __init__(self, vocab_size: int, n_blocks: int, d_model: int, 
                 n_heads: int , d_ff: int , dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded_text = self.embeddings(text)
        embedded_text = self.pos_encoding(embedded_text)
        transformer_output = self.transformer_blocks(embedded_text)
        pooled_output = transformer_output.mean(axis=1)
        logits = self.fc(pooled_output)
        return logits.squeeze(-1)


def VanillaBert(vocab_size, **kwargs):
    ''' Default parameters for the vanilla BERT '''
    params = {
        'n_blocks': 12, 
        'd_model': 768, 
        'n_heads': 12, 
        'd_ff': 768 * 4,
        'dropout': 0.1,
        'max_len': 128,
    }
    params.update(kwargs)
    return TransformerModule(vocab_size=vocab_size, **params)