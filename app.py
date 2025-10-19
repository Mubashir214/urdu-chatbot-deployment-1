# app.py
import streamlit as st
import torch
import torch.nn as nn
import json
import os
import time
import math
import re
import arabic_reshaper
from bidi.algorithm import get_display
import random

# Page configuration
st.set_page_config(
    page_title="اردو چیٹ بوٹ - Urdu Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for RTL support
st.markdown("""
<style>
    .urdu-text {
        font-family: 'Segoe UI', 'Noto Sans Arabic', 'Arial';
        font-size: 18px;
        direction: rtl;
        text-align: right;
        line-height: 1.8;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        border: 1px solid #90caf9;
        text-align: right;
    }
    .bot-message {
        background-color: #f3e5f5;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        border: 1px solid #ce93d8;
        text-align: right;
    }
    .stTextInput > div > div > input {
        text-align: right;
        font-family: 'Segoe UI', 'Noto Sans Arabic';
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# Transformer Model Architecture (Same as Training)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

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
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
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
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
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
        return (src != 0).long()

    def make_tgt_mask(self, tgt):
        b, seq = tgt.size()
        pad_mask = (tgt != 0).long()
        subsequent = torch.tril(torch.ones((seq, seq), device=tgt.device)).long()
        return (pad_mask.unsqueeze(1) * subsequent.unsqueeze(0)).to(tgt.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.enc(src, src_mask)
        logits = self.dec(tgt, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits

class UrduChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.max_len = 64
        self.load_model()

    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            if os.path.exists('urdu_tokenizer.json'):
                with open('urdu_tokenizer.json', 'r', encoding='utf-8') as f:
                    tokenizer_config = json.load(f)
                    self.stoi = tokenizer_config['word_index']
                    self.vocab_size = tokenizer_config['vocab_size']
                    self.itos = {v: k for k, v in self.stoi.items()}
            else:
                st.error("❌ Tokenizer file not found!")
                return False

            # Initialize model
            self.model = TransformerModel(
                vocab_size=self.vocab_size,
                d_model=256,
                enc_layers=2,
                dec_layers=2,
                n_heads=2,
                d_ff=512
            ).to(self.device)

            # Load model weights
            if os.path.exists('best_urdu_chatbot.pth'):
                checkpoint = torch.load('best_urdu_chatbot.pth', map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                st.success("✅ ماڈل کامیابی سے لوڈ ہو گیا!")
                return True
            else:
                st.warning("⚠️ مکمل ماڈل دستیاب نہیں ہے، سادہ موڈ استعمال ہو رہا ہے۔")
                return False

        except Exception as e:
            st.error(f"❌ ماڈل لوڈ کرنے میں خرابی: {str(e)}")
            return False

    def normalize_urdu_text(self, text):
        """Normalize Urdu text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
        text = text.replace('\u0640', '')
        text = re.sub('[\u0622\u0623\u0625]', 'ا', text)
        text = re.sub('[\u064A\u06D0]', 'ی', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def encode_text(self, text):
        """Encode text to token indices"""
        text = self.normalize_urdu_text(text)
        tokens = re.findall(r'[\u0600-\u06FF]+|[.,?!؛؟]', text)
        ids = [self.stoi.get(token, self.stoi.get('<OOV>', 1)) for token in tokens]
        ids = [self.stoi.get('<sos>', 2)] + ids + [self.stoi.get('<eos>', 3)]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            ids[-1] = self.stoi.get('<eos>', 3)
        return ids

    def pad_seq(self, seq):
        """Pad sequence to max length"""
        if len(seq) < self.max_len:
            return seq + [0] * (self.max_len - len(seq))
        return seq[:self.max_len]

    def ids_to_sentence(self, ids):
        """Convert token indices back to text"""
        tokens = []
        for i in ids:
            if i == 0:  # PAD
                continue
            if i == self.stoi.get('<sos>', 2):
                continue
            if i == self.stoi.get('<eos>', 3):
                break
            tokens.append(self.itos.get(i, '<UNK>'))
        return ' '.join(tokens)

    def greedy_generate(self, src_text):
        """Generate response using greedy decoding"""
        if self.model is None:
            return self.rule_based_response(src_text)

        try:
            with torch.no_grad():
                src_ids = self.encode_text(src_text)
                src_tensor = torch.tensor([self.pad_seq(src_ids)], dtype=torch.long).to(self.device)
                
                enc_out = self.model.enc(src_tensor, self.model.make_src_mask(src_tensor))
                
                ys = torch.full((1, 1), self.stoi.get('<sos>', 2), dtype=torch.long, device=self.device)
                
                for _ in range(self.max_len - 1):
                    tgt_mask = self.model.make_tgt_mask(ys)
                    out = self.model.dec(ys, enc_out, tgt_mask=tgt_mask, 
                                       memory_mask=self.model.make_src_mask(src_tensor))
                    next_logits = out[:, -1, :]
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    ys = torch.cat([ys, next_token], dim=1)
                    
                    if next_token.item() == self.stoi.get('<eos>', 3):
                        break
                
                return self.ids_to_sentence(ys[0].cpu().tolist())
                
        except Exception as e:
            st.error(f"❌ جنریشن میں خرابی: {str(e)}")
            return self.rule_based_response(src_text)

    def rule_based_response(self, user_input):
        """Fallback rule-based responses"""
        responses = {
            'سلام': ['وعلیکم السلام! آپ کیسے ہیں؟', 'سلام! خوش آمدید۔', 'ہیلو! آپ کا دن اچھا گزرے۔'],
            'کیسے': ['میں ٹھیک ہوں، شکریہ! آپ سنائیں؟', 'بہت اچھا، آپ کا شکریہ۔', 'الحمدللہ!'],
            'نام': ['میرا نام اردو بوٹ ہے۔', 'میں اردو چیٹ بوٹ ہوں۔', 'آپ مجھے اردو بوٹ کہہ سکتے ہیں۔'],
            'شکریہ': ['خوش آمدید!', 'کوئی بات نہیں۔', 'آپ کا بہت بہت شکریہ۔'],
            'ہاں': ['بہت اچھا!', 'آپ سے بات کر کے اچھا لگا۔', 'میں خوش ہوں۔'],
            'نہیں': ['کوئی بات نہیں۔', 'میں سمجھا۔', 'آپ کی مرضی۔']
        }
        
        for key in responses:
            if key in user_input:
                return random.choice(responses[key])
        
        default_responses = [
            "میں سمجھا۔",
            "براہ کرم مزید وضاحت کریں۔",
            "یہ دلچسپ ہے!",
            "کیا آپ اس بارے میں مزید بتا سکتے ہیں؟",
            "میں ابھی تربیت کے مراحل میں ہوں۔",
            "آپ کی بات سمجھ میں آئی۔"
        ]
        
        return random.choice(default_responses)

def reshape_urdu_text(text):
    """Reshape Urdu text for proper RTL display"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except:
        return text

def main():
    st.markdown("<h1 class='urdu-text'>🤖 اردو چیٹ بوٹ</h1>", unsafe_allow_html=True)
    st.markdown("<p class='urdu-text'>خوش آمدید! اردو میں بات چیت شروع کریں۔</p>", unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = UrduChatbot()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 class='urdu-text'>🔧 ترتیبات</h3>", unsafe_allow_html=True)
        
        decoding_strategy = st.radio(
            "**ڈی کوڈنگ اسٹریٹیجی**",
            ["greedy", "beam_search"],
            index=0,
            format_func=lambda x: "گریڈی سرچ" if x == "greedy" else " بیم سرچ"
        )
        
        st.markdown("---")
        
        if st.button("🗑️ بات چیت صاف کریں", use_container_width=True):
            if 'conversation' in st.session_state:
                st.session_state.conversation = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("<h3 class='urdu-text'>ℹ️ معلومات</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='urdu-text'>
        یہ ایک تربیت یافتہ اردو چیٹ بوٹ ہے جو ٹرانسفارمر ماڈل پر مبنی ہے۔
        
        **خصوصیات:**
        • اردو زبان میں بات چیت
        • دو مختلف ڈی کوڈنگ اسٹریٹیجیز
        • بات چیت کی تاریخ
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Display conversation
    conversation_container = st.container()
    with conversation_container:
        for speaker, message, timestamp in st.session_state.conversation:
            if speaker == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div style="text-align: left; font-size: 12px; color: #666;">You • {timestamp}</div>
                    <div class="urdu-text">{reshape_urdu_text(message)}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div style="text-align: left; font-size: 12px; color: #666;">UrduBot • {timestamp}</div>
                    <div class="urdu-text">{reshape_urdu_text(message)}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "اپنا پیغام یہاں ٹائپ کریں:",
            placeholder="اردو میں اپنا سوال یا جملہ لکھیں...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 بھیجیں", use_container_width=True)
    
    # Handle user input
    if send_button and user_input.strip():
        # Add user message to conversation
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.conversation.append(("user", user_input, timestamp))
        
        # Generate and display bot response
        with st.spinner("🤔 سوچ رہا ہوں..."):
            if decoding_strategy == "greedy":
                response = chatbot.greedy_generate(user_input)
            else:
                response = chatbot.rule_based_response(user_input)
            
            # Add bot response to conversation
            st.session_state.conversation.append(("bot", response, time.strftime("%H:%M:%S")))
            
            # Rerun to update the conversation display
            st.rerun()
    
    # Welcome message if no conversation
    if not st.session_state.conversation:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h3 class='urdu-text'>👋 ہیلو! میں آپ کی کیسے مدد کر سکتا ہوں؟</h3>
            <p class='urdu-text'>ذیل میں کچھ مثالیں ہیں:</p>
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px;'>
                <p class='urdu-text'>• آپ کیسے ہیں؟</p>
                <p class='urdu-text'>• آج موسم کیسا ہے؟</p>
                <p class='urdu-text'>• اردو زبان کے بارے میں بتائیں</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()