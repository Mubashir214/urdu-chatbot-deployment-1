# app.py - WITH MODEL LOADING
import streamlit as st
import torch
import torch.nn as nn
import json
import re
import math
import time
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
        font-family: 'Segoe UI', 'Arial', sans-serif;
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
        font-family: 'Segoe UI', 'Arial';
        font-size: 16px;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Transformer Model Definition
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

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, enc_layers=2, dec_layers=2, n_heads=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.enc_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.enc_pos = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
            for _ in range(enc_layers)
        ])
        
        # Decoder
        self.dec_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dec_pos = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
            for _ in range(dec_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        return (src != 0)

    def make_tgt_mask(self, tgt_len, device):
        mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, src, tgt):
        # Encoder
        src_mask = self.make_src_mask(src)
        src_embedded = self.enc_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.enc_pos(src_embedded)
        
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=~src_mask)
        
        # Decoder
        tgt_mask = self.make_tgt_mask(tgt.size(1), tgt.device)
        tgt_embedded = self.dec_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.dec_pos(tgt_embedded)
        
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, 
                          tgt_key_padding_mask=~self.make_src_mask(tgt),
                          memory_key_padding_mask=~src_mask)
        
        return self.output_layer(output)

class UrduChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.max_len = 50
        self.load_model()

    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            if not os.path.exists('urdu_tokenizer.json'):
                return False
                
            with open('urdu_tokenizer.json', 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
                self.stoi = tokenizer_config['word_index']
                self.vocab_size = tokenizer_config['vocab_size']
                self.itos = {v: k for k, v in self.stoi.items()}
            
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
            if not os.path.exists('best_urdu_chatbot.pth'):
                return False
                
            checkpoint = torch.load('best_urdu_chatbot.pth', map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading error: {str(e)}")
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
        
        # Convert tokens to IDs
        ids = [self.stoi.get(token, self.stoi.get('<OOV>', 1)) for token in tokens]
        
        # Add SOS and EOS
        sos_token = self.stoi.get('<sos>', 2)
        eos_token = self.stoi.get('<eos>', 3)
        ids = [sos_token] + ids + [eos_token]
        
        # Truncate or pad
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            ids[-1] = eos_token
        else:
            ids.extend([0] * (self.max_len - len(ids)))
            
        return torch.tensor([ids], dtype=torch.long)

    def decode_text(self, ids):
        """Convert token indices back to text"""
        tokens = []
        for id_val in ids:
            if id_val == 0:  # PAD
                continue
            if id_val == self.stoi.get('<sos>', 2):
                continue
            if id_val == self.stoi.get('<eos>', 3):
                break
            tokens.append(self.itos.get(id_val, '<UNK>'))
        return ' '.join(tokens)

    def generate_response(self, user_input):
        """Generate response using the trained model"""
        try:
            if self.model is None:
                return self.fallback_response(user_input)
                
            with torch.no_grad():
                # Encode input
                src_tensor = self.encode_text(user_input).to(self.device)
                
                # Start with SOS token
                batch_size = src_tensor.size(0)
                tgt_tensor = torch.full((batch_size, 1), self.stoi.get('<sos>', 2), 
                                      dtype=torch.long, device=self.device)
                
                # Generate tokens one by one
                for i in range(self.max_len - 1):
                    output = self.model(src_tensor, tgt_tensor)
                    next_token_logits = output[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    tgt_tensor = torch.cat([tgt_tensor, next_token], dim=1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == self.stoi.get('<eos>', 3):
                        break
                
                # Decode the generated tokens
                response = self.decode_text(tgt_tensor[0].cpu().numpy())
                return response if response else self.fallback_response(user_input)
                
        except Exception as e:
            st.error(f"❌ Generation error: {str(e)}")
            return self.fallback_response(user_input)

    def fallback_response(self, user_input):
        """Fallback responses if model fails"""
        fallback_responses = {
            'سلام': ['وعلیکم السلام! آپ کیسے ہیں؟', 'سلام! خوش آمدید۔'],
            'کیسے': ['میں ٹھیک ہوں، شکریہ! آپ سنائیں؟'],
            'نام': ['میرا نام اردو بوٹ ہے۔'],
            'شکریہ': ['خوش آمدید!'],
            'ہاں': ['بہت اچھا!'],
            'نہیں': ['کوئی بات نہیں۔']
        }
        
        for key in fallback_responses:
            if key in user_input.lower():
                return random.choice(fallback_responses[key])
        
        return random.choice([
            "میں سمجھا۔",
            "براہ کرم مزید وضاحت کریں۔",
            "یہ دلچسپ ہے!",
            "کیا آپ اس بارے میں مزید بتا سکتے ہیں؟"
        ])

def main():
    st.markdown("<h1 class='urdu-text'>🤖 اردو چیٹ بوٹ</h1>", unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = UrduChatbot()
    
    # Show model status
    if chatbot.model is not None:
        st.markdown("<div class='success-box'>✅ ماڈل کامیابی سے لوڈ ہو گیا!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='error-box'>⚠️ ماڈل لوڈ نہیں ہو سکا، فال بیک موڈ استعمال ہو رہا ہے</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='urdu-text'>خوش آمدید! اردو میں بات چیت شروع کریں۔</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 class='urdu-text'>🔧 ترتیبات</h3>", unsafe_allow_html=True)
        
        decoding = st.radio(
            "**ڈی کوڈنگ**",
            ["گریڈی", "بیم سرچ"],
            index=0
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
        تربیت یافتہ اردو چیٹ بوٹ
        
        **خصوصیات:**
        • ٹرانسفارمر ماڈل
        • اردو زبان
        • فطری جوابات
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Display conversation
    for speaker, message, timestamp in st.session_state.conversation:
        if speaker == "user":
            st.markdown(f"""
            <div class="user-message">
                <div style="text-align: left; font-size: 12px; color: #666;">You • {timestamp}</div>
                <div class="urdu-text">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <div style="text-align: left; font-size: 12px; color: #666;">UrduBot • {timestamp}</div>
                <div class="urdu-text">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "اپنا پیغام یہاں ٹائپ کریں:",
            placeholder="اردو میں اپنا سوال لکھیں...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 بھیجیں", use_container_width=True)
    
    # Handle user input
    if send_button and user_input.strip():
        # Add user message
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.conversation.append(("user", user_input, timestamp))
        
        # Generate response
        with st.spinner("🤔 سوچ رہا ہوں..."):
            response = chatbot.generate_response(user_input)
            
            # Add bot response
            st.session_state.conversation.append(("bot", response, time.strftime("%H:%M:%S")))
            st.rerun()
    
    # Welcome message
    if not st.session_state.conversation:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h3 class='urdu-text'>👋 ہیلو! میں آپ کی کیسے مدد کر سکتا ہوں؟</h3>
            <p class='urdu-text'>مثالیں:</p>
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px;'>
                <p class='urdu-text'>• آپ کیسے ہیں؟</p>
                <p class='urdu-text'>• آپ کا نام کیا ہے؟</p>
                <p class='urdu-text'>• آج موسم کیسا ہے؟</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
