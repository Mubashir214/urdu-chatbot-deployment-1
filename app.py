# app.py
import streamlit as st
import torch
import json
import os
import time
import re
import random

# Try to import RTL libraries, but provide fallback
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_SUPPORT = True
except ImportError:
    RTL_SUPPORT = False
    st.warning("RTL libraries not available, using basic Urdu display")

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
</style>
""", unsafe_allow_html=True)

def reshape_urdu_text(text):
    """Reshape Urdu text for proper RTL display"""
    if not RTL_SUPPORT:
        return text
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except:
        return text

class SimpleUrduChatbot:
    def __init__(self):
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Try to load the model, but provide fallback"""
        try:
            # Check if model files exist
            if os.path.exists('best_urdu_chatbot.pth') and os.path.exists('urdu_tokenizer.json'):
                st.success("✅ ماڈل فائلیں موجود ہیں!")
                self.model_loaded = True
            else:
                st.warning("⚠️ مکمل ماڈل دستیاب نہیں ہے، سادہ موڈ استعمال ہو رہا ہے۔")
                self.model_loaded = False
        except Exception as e:
            st.error(f"❌ ماڈل لوڈ کرنے میں خرابی: {str(e)}")
            self.model_loaded = False
    
    def generate_response(self, user_input):
        """Generate response based on user input"""
        return self.rule_based_response(user_input)
    
    def rule_based_response(self, user_input):
        """Rule-based responses for demo"""
        responses = {
            'سلام': ['وعلیکم السلام! آپ کیسے ہیں؟', 'سلام! خوش آمدید۔', 'ہیلو! آپ کا دن اچھا گزرے۔'],
            'کیسے': ['میں ٹھیک ہوں، شکریہ! آپ سنائیں؟', 'بہت اچھا، آپ کا شکریہ۔', 'الحمدللہ!'],
            'نام': ['میرا نام اردو بوٹ ہے۔', 'میں اردو چیٹ بوٹ ہوں۔', 'آپ مجھے اردو بوٹ کہہ سکتے ہیں۔'],
            'شکریہ': ['خوش آمدید!', 'کوئی بات نہیں۔', 'آپ کا بہت بہت شکریہ۔'],
            'ہاں': ['بہت اچھا!', 'آپ سے بات کر کے اچھا لگا۔', 'میں خوش ہوں۔'],
            'نہیں': ['کوئی بات نہیں۔', 'میں سمجھا۔', 'آپ کی مرضی۔'],
            'کیا': ['جی ہاں؟', 'کیا بات ہے؟', 'میں سن رہا ہوں۔'],
            'کون': ['میں ایک چیٹ بوٹ ہوں۔', 'میں مصنوعی ذہانت کا پروگرام ہوں۔'],
            'کہاں': ['میں آن لائن موجود ہوں۔', 'میں ہر جگہ موجود ہوں۔'],
            'کیوں': ['کیونکہ ایسا ہی ہوتا ہے۔', 'یہ ایک اچھا سوال ہے۔']
        }
        
        # Find matching response
        user_input_lower = user_input.lower()
        for key in responses:
            if key in user_input_lower:
                return random.choice(responses[key])
        
        # Default responses
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
    st.markdown("<p class='urdu-text'>خوش آمدید! اردو میں بات چیت شروع کریں۔</p>", unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = SimpleUrduChatbot()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3 class='urdu-text'>🔧 ترتیبات</h3>", unsafe_allow_html=True)
        
        # Simple settings
        st.radio(
            "**موڈ**",
            ["سادہ", "اعلیٰ"],
            index=0,
            key="mode"
        )
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ بات چیت صاف کریں", use_container_width=True):
            if 'conversation' in st.session_state:
                st.session_state.conversation = []
            st.rerun()
        
        st.markdown("---")
        
        # Information section
        st.markdown("<h3 class='urdu-text'>ℹ️ معلومات</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='urdu-text'>
        یہ ایک اردو چیٹ بوٹ ہے جو مصنوعی ذہانت پر مبنی ہے۔
        
        **خصوصیات:**
        • اردو زبان میں بات چیت
        • فطری جوابات
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
            response = chatbot.generate_response(user_input)
            
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
                <p class='urdu-text'>• آپ کا نام کیا ہے؟</p>
                <p class='urdu-text'>• آج موسم کیسا ہے؟</p>
                <p class='urdu-text'>• اردو زبان کے بارے میں بتائیں</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
