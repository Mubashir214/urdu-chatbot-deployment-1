# app.py - COMPLETE WORKING VERSION
import streamlit as st
import torch
import torch.nn as nn
import json
import re
import math
import time
import random
import os

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
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Simple Transformer Model (Compatible with your trained model)
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = self.create_pos_encoding(5000, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def make_src_mask(self, src):
        return (src != 0)
    
    def make_tgt_mask(self, tgt_len, device):
        mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
        return mask.to(device)
    
    def forward(self, src, tgt):
        # Source processing
        src_mask = self.make_src_mask(src)
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_embedded = src_embedded + self.pos_encoding[:, :src.size(1), :].to(src.device)
        
        memory = self.encoder(src_embedded, src_key_padding_mask=~src_mask)
        
        # Target processing
        tgt_mask = self.make_tgt_mask(tgt.size(1), tgt.device)
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask, 
                            tgt_key_padding_mask=~self.make_src_mask(tgt),
                            memory_key_padding_mask=~src_mask)
        
        return self.output_layer(output)

class UrduChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 10000  # Default, will be updated from tokenizer
        self.max_len = 50
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            st.info("🔄 ماڈل لوڈ ہو رہا ہے...")
            
            # Check if files exist
            if not os.path.exists('urdu_tokenizer.json'):
                st.error("❌ urdu_tokenizer.json فائل نہیں ملی")
                return False
                
            if not os.path.exists('best_urdu_chatbot.pth'):
                st.error("❌ best_urdu_chatbot.pth فائل نہیں ملی")
                return False
            
            # Load tokenizer
            with open('urdu_tokenizer.json', 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
                self.stoi = tokenizer_config['word_index']
                self.vocab_size = tokenizer_config.get('vocab_size', len(self.stoi) + 4)
                self.itos = {v: k for k, v in self.stoi.items()}
            
            st.success(f"✅ لغت لوڈ ہو گئی: {len(self.stoi)} الفاظ")
            
            # Initialize model
            self.model = SimpleTransformer(
                vocab_size=self.vocab_size,
                d_model=256,
                nhead=8,
                num_layers=2
            ).to(self.device)
            
            # Load model weights
            checkpoint = torch.load('best_urdu_chatbot.pth', map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try loading the state dict directly
                    self.model.load_state_dict(checkpoint)
            else:
                # If it's directly the state dict
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            st.success("✅ ماڈل کامیابی سے لوڈ ہو گیا!")
            return True
            
        except Exception as e:
            st.error(f"❌ ماڈل لوڈ کرنے میں خرابی: {str(e)}")
            import traceback
            st.error(f"تفصیلی خرابی: {traceback.format_exc()}")
            return False

    def normalize_urdu_text(self, text):
        """Normalize Urdu text"""
        if not isinstance(text, str):
            return ""
        # Remove diacritics and normalize characters
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
        text = text.replace('\u0640', '')  # Remove Tatweel
        text = re.sub('[\u0622\u0623\u0625]', 'ا', text)  # Normalize Alef
        text = re.sub('[\u064A\u06CC]', 'ی', text)  # Normalize Yeh
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def encode_text(self, text):
        """Encode text to token indices"""
        try:
            text = self.normalize_urdu_text(text)
            if not text:
                return torch.tensor([[2, 3] + [0] * (self.max_len-2)], dtype=torch.long)
            
            # Simple tokenization (split by spaces)
            tokens = text.split()
            
            # Convert tokens to IDs
            ids = []
            for token in tokens:
                if token in self.stoi:
                    ids.append(self.stoi[token])
                else:
                    ids.append(self.stoi.get('<OOV>', 1))  # Use OOV token
            
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
            
        except Exception as e:
            st.error(f"Encoding error: {e}")
            # Return basic SOS-EOS sequence
            return torch.tensor([[2, 3] + [0] * (self.max_len-2)], dtype=torch.long)

    def decode_text(self, ids):
        """Convert token indices back to text"""
        try:
            tokens = []
            for id_val in ids:
                if id_val == 0:  # PAD token
                    continue
                if id_val == self.stoi.get('<sos>', 2):  # SOS token
                    continue
                if id_val == self.stoi.get('<eos>', 3):  # EOS token
                    break
                token = self.itos.get(id_val, '<UNK>')
                tokens.append(token)
            return ' '.join(tokens) if tokens else "مجھے سمجھ نہیں آیا۔"
        except Exception as e:
            return "جواب جنریٹ کرنے میں مسئلہ ہوا۔"

    def generate_response(self, user_input):
        """Generate response using the trained model"""
        try:
            if not self.model_loaded:
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
                return response if response.strip() else self.fallback_response(user_input)
                
        except Exception as e:
            st.error(f"❌ جنریشن میں خرابی: {str(e)}")
            return self.fallback_response(user_input)

    def fallback_response(self, user_input):
        """Fallback responses if model fails"""
        fallback_responses = {
            'سلام': ['وعلیکم السلام! آپ کیسے ہیں؟', 'سلام! خوش آمدید۔', 'ہیلو! آپ کا دن اچھا گزرے۔'],
            'ہیلو': ['ہیلو! خوش آمدید۔', 'ہیلو! آپ کیسے ہیں؟'],
            'آداب': ['آداب! بہت خوشی ہوئی۔', 'آداب! آپ کیسے ہیں؟'],
            'کیسے': ['میں ٹھیک ہوں، شکریہ! آپ سنائیں؟', 'بہت اچھا، آپ کا شکریہ۔', 'الحمدللہ!'],
            'حال': ['میرا حال اچھا ہے، آپ کا شکریہ۔', 'بہت اچھا، آپ کا حال کیسا ہے؟'],
            'نام': ['میرا نام اردو بوٹ ہے۔', 'میں اردو چیٹ بوٹ ہوں۔', 'آپ مجھے اردو بوٹ کہہ سکتے ہیں۔'],
            'شکریہ': ['خوش آمدید!', 'کوئی بات نہیں۔', 'آپ کا بہت بہت شکریہ۔'],
            'شکریا': ['خوش آمدید!', 'کوئی بات نہیں دوست۔'],
            'ہاں': ['بہت اچھا!', 'آپ سے بات کر کے اچھا لگا۔', 'میں خوش ہوں۔'],
            'نہیں': ['کوئی بات نہیں۔', 'میں سمجھا۔', 'آپ کی مرضی۔'],
            'جی': ['جی ہاں!', 'بالکل!', 'آپ درست کہہ رہے ہیں۔'],
            'جی ہاں': ['بہت خوب!', 'میں متفق ہوں۔', 'یہ اچھی بات ہے۔'],
            'کیا': ['جی ہاں؟', 'کیا بات ہے؟', 'میں سن رہا ہوں۔'],
            'کون': ['میں ایک چیٹ بوٹ ہوں۔', 'میں مصنوعی ذہانت کا پروگرام ہوں۔'],
            'کہاں': ['میں آن لائن موجود ہوں۔', 'میں ہر جگہ موجود ہوں۔'],
            'کیوں': ['کیونکہ ایسا ہی ہوتا ہے۔', 'یہ ایک اچھا سوال ہے۔'],
            'کب': ['جلد ہی۔', 'ابھی۔', 'مستقبل قریب میں۔'],
            'موسم': ['آج موسم بہت خوبصورت ہے۔', 'موسم خوشگوار ہے۔'],
            'وقت': ['ابھی بات چیت کا وقت ہے۔', 'ہمیشہ آپ سے بات کرنے کا اچھا وقت ہے۔'],
            'مدد': ['میں آپ کی کیسے مدد کر سکتا ہوں؟', 'براہ کرم بتائیں آپ کو کس چیز میں مدد چاہیے؟'],
            'اردو': ['اردو بہت خوبصورت زبان ہے۔', 'میں اردو بول سکتا ہوں۔'],
            'زبان': ['زبان رابطے کا ذریعہ ہے۔', 'اردو میری پسندیدہ زبان ہے۔'],
            'خوش': ['آپ کو خوش دیکھ کر اچھا لگا۔', 'یہ بہت اچھی بات ہے۔'],
            'غم': ['غمگسار ہوں۔', 'کچھ پریشانی ہے؟'],
            'خدا حافظ': ['خدا حافظ! آپ کا دن اچھا گزرے۔', 'الوداع! پھر ملتے ہیں۔'],
            'الوداع': ['الوداع!', 'خدا حافظ دوست!']
        }
        
        user_input_lower = user_input.lower()
        for key in fallback_responses:
            if key in user_input_lower:
                return random.choice(fallback_responses[key])
        
        default_responses = [
            "میں سمجھا۔",
            "براہ کرم مزید وضاحت کریں۔",
            "یہ دلچسپ ہے!",
            "کیا آپ اس بارے میں مزید بتا سکتے ہیں؟",
            "میں ابھی تربیت کے مراحل میں ہوں۔",
            "آپ کی بات سمجھ میں آئی۔",
            "مجھے اردو سیکھنے میں مدد کریں۔",
            "کیا آپ کو اردو بولنا آتا ہے؟",
            "یہ ایک خوبصورت زبان ہے۔",
            "میں آپ کی مدد کے لیے یہاں ہوں۔"
        ]
        
        return random.choice(default_responses)

def main():
    st.markdown("<h1 class='urdu-text'>🤖 اردو چیٹ بوٹ</h1>", unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = UrduChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Show model status
    if chatbot.model_loaded:
        st.markdown("<div class='success-box'>✅ ماڈل کامیابی سے لوڈ ہو گیا!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>⚠️ ماڈل لوڈ نہیں ہو سکا، فال بیک موڈ استعمال ہو رہا ہے</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='urdu-text'>خوش آمدید! اردو میں بات چیت شروع کریں۔</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 class='urdu-text'>🔧 ترتیبات</h3>", unsafe_allow_html=True)
        
        # Model info
        if chatbot.model_loaded:
            st.success(f"ماڈل: {chatbot.vocab_size} الفاظ")
        else:
            st.warning("سادہ موڈ")
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("🗑️ بات چیت صاف کریں", use_container_width=True):
            if 'conversation' in st.session_state:
                st.session_state.conversation = []
            st.rerun()
        
        # Reload model
        if st.button("🔄 ماڈل دوبارہ لوڈ کریں", use_container_width=True):
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot
            st.rerun()
        
        st.markdown("---")
        st.markdown("<h3 class='urdu-text'>ℹ️ معلومات</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='urdu-text'>
        **اردو چیٹ بوٹ**
        
        تربیت یافتہ ٹرانسفارمر ماڈل
        
        استعمال:
        1. اردو میں لکھیں
        2. بھیجیں پر کلک کریں  
        3. جواب پائیں
        
        مثالیں:
        • سلام
        • آپ کیسے ہیں؟
        • آپ کا نام کیا ہے؟
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
    st.markdown("### 💭 نیا پیغام")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "اپنا پیغام یہاں ٹائپ کریں:",
            placeholder="مثال: آپ کیسے ہیں؟",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 بھیجیں", use_container_width=True)
    
    # Quick example buttons
    st.markdown("**فوری مثالیں:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("👋 سلام", use_container_width=True, key="btn1"):
            user_input = "سلام"
    with col2:
        if st.button("❓ کیسے ہیں", use_container_width=True, key="btn2"):
            user_input = "آپ کیسے ہیں"
    with col3:
        if st.button("ℹ️ نام", use_container_width=True, key="btn3"):
            user_input = "آپ کا نام کیا ہے"
    with col4:
        if st.button("🌤️ موسم", use_container_width=True, key="btn4"):
            user_input = "آج موسم کیسا ہے"
    
    # Handle user input
    if send_button and user_input.strip():
        # Add user message to conversation
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.conversation.append(("user", user_input, timestamp))
        
        # Generate and display bot response
        with st.spinner("🤔 سوچ رہا ہوں..."):
            response = chatbot.generate_response(user_input)
            
            # Add bot response to conversation
            st.session_state.conversation.append(("bot", response, time.strftime("%H:%M:%S")))
            
            # Rerun to update the conversation display
            st.rerun()
    
    # Welcome message if no conversation
    if not st.session_state.conversation:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background-color: #f0f2f6; border-radius: 10px;'>
            <h3 class='urdu-text'>👋 ہیلو! میں آپ کی کیسے مدد کر سکتا ہوں؟</h3>
            <p class='urdu-text'>اوپر دیے گئے بٹن استعمال کریں یا خود لکھیں!</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>🤖 اردو چیٹ بوٹ - Built with Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
