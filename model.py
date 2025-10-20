# model.py
import torch
import torch.nn as nn
import math
import re
from typing import List, Tuple, Dict
import os

# Constants - MUST match your training
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

# Your exact model architecture from training
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, d = x.size()
        x = x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        return x

    def combine_heads(self, x):
        b, h, s, dk = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, h * dk)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.q_lin(q))
        K = self.split_heads(self.k_lin(k))
        V = self.split_heads(self.v_lin(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                mask_ = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask_ = mask.unsqueeze(1)
            else:
                mask_ = mask
            scores = scores.masked_fill(mask_ == 0, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = self.combine_heads(out)
        return self.out_lin(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        sa = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(sa))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        sa = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa))
        ca = self.cross_attn(x, enc_out, enc_out, mask=memory_mask)
        x = self.norm2(x + self.dropout(ca))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_emb(src) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        x = self.norm(x)
        return self.out(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, enc_layers=2, dec_layers=2, n_heads=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.enc = Encoder(vocab_size, d_model, enc_layers, n_heads, d_ff, dropout)
        self.dec = Decoder(vocab_size, d_model, dec_layers, n_heads, d_ff, dropout)

    def make_src_mask(self, src):
        return (src != PAD_IDX).long()

    def make_tgt_mask(self, tgt):
        b, seq = tgt.size()
        pad_mask = (tgt != PAD_IDX).long()
        subsequent = torch.tril(torch.ones((seq, seq), device=tgt.device)).long()
        return (pad_mask.unsqueeze(1) * subsequent.unsqueeze(0)).to(tgt.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.enc(src, src_mask)
        logits = self.dec(tgt, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits

# Vocabulary and text processing utilities
def load_vocab(vocab_path="vocab.txt"):
    """Load vocabulary from file - matches your training format"""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file {vocab_path} not found")
    
    itos = [PAD, SOS, EOS, UNK]
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                word = parts[0]
                if word and word not in itos:
                    itos.append(word)
    
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def normalize_urdu(text):
    """Your exact normalization function from training"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
    text = text.replace('\u0640', '')
    text = re.sub('[\u0622\u0623\u0625]', 'ا', text)
    text = re.sub('[\u064A\u06D0]', 'ی', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_text(text: str, stoi, max_len=64):
    """Encode text to token IDs - matches your training"""
    text = normalize_urdu(text)
    tokens = text.split()
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    if len(ids) > max_len:
        ids = ids[:max_len]
        if ids[-1] != EOS_IDX:
            ids[-1] = EOS_IDX
    return ids

def pad_seq(seq, max_len=64, pad_idx=0):
    """Pad sequence to fixed length"""
    if len(seq) < max_len:
        return seq + [pad_idx] * (max_len - len(seq))
    return seq[:max_len]

def ids_to_sentence(ids: List[int], itos):
    """Convert token IDs back to text"""
    tokens = []
    for i in ids:
        if i == PAD_IDX:
            continue
        if i == SOS_IDX:
            continue
        if i == EOS_IDX:
            break
        tokens.append(itos[i] if i < len(itos) else UNK)
    return " ".join(tokens)

# Model loading and generation
def load_model(model_path="best_transformer_bleu.pt", vocab_path="vocab.txt"):
    """Load trained model and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary first
    stoi, itos = load_vocab(vocab_path)
    vocab_size = len(itos)
    
    # Set global indices
    global PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
    PAD_IDX = stoi[PAD]
    SOS_IDX = stoi[SOS]
    EOS_IDX = stoi[EOS]
    UNK_IDX = stoi[UNK]
    
    # Initialize model with correct architecture
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=256,
        enc_layers=2,
        dec_layers=2,
        n_heads=2,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model file {model_path} not found. Using untrained model.")
    
    model.eval()
    return model, stoi, itos, device

def generate_response(model, text, stoi, itos, device, max_len=64):
    """Generate response for given text"""
    model.eval()
    with torch.no_grad():
        # Encode input
        src_ids = pad_seq(encode_text(text, stoi, max_len))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        # Encode source
        src_mask = model.make_src_mask(src_tensor)
        enc_out = model.enc(src_tensor, src_mask)
        
        # Start with SOS token
        ys = torch.full((1, 1), SOS_IDX, dtype=torch.long, device=device)
        
        # Generate tokens
        for _ in range(max_len - 1):
            tgt_mask = model.make_tgt_mask(ys)
            out = model.dec(ys, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask)
            next_logits = out[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == EOS_IDX:
                break
        
        # Convert back to text
        return ids_to_sentence(ys[0].cpu().tolist(), itos)

# Initialize globals (will be set when model is loaded)
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3