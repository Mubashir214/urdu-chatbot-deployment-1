# app.py
import streamlit as st
import torch
import time
import random
from model import load_model, generate_response

# Page configuration
st.set_page_config(
    page_title="اردو چیٹ بوٹ | Urdu Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Urdu support
st.markdown("""
<style>
    .urdu-text {
        font-family: 'Segoe UI', 'Nafees Regular', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', Tahoma, Geneva, Verdana, sans-serif;
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
    }
    .bot-message {
        background-color: #f3e5f5;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        border: 1px solid #ce93d8;
    }
    .title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 80%;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'model_components' not in st.session_state:
        st.session_state.model_components = None

def load_chatbot_model():
    """Load the chatbot model with caching"""
    if not st.session_state.model_loaded:
        with st.spinner("🔄 اردو چیٹ بوٹ لوڈ ہو رہا ہے... براہ کرم انتظار کریں"):
            try:
                # Try to load your actual trained model
                model, stoi, itos, device = load_model(
                    model_path="best_transformer_bleu.pt",
                    vocab_path="vocab.txt"
                )
                st.session_state.model_components = (model, stoi, itos, device)
                st.session_state.model_loaded = True
                st.success("✅ چیٹ بوٹ کامیابی سے لوڈ ہو گیا!")
                return True
            except Exception as e:
                st.error(f"❌ ماڈل لوڈ کرنے میں مسئلہ: {e}")
                return False
    return True

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="title">🤖 اردو چیٹ بوٹ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Urdu Language AI Chatbot - Transformers from Scratch</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ معلومات / Information")
        st.markdown("""
        **مثالی سوالات / Example Questions:**
        - آپ کیسے ہیں؟
        - آپ کا نام کیا ہے؟
        - آج موسم کیسا ہے؟
        - اردو سیکھنے کے لیے تجاویز
        - کیا آپ مجھے مدد کر سکتے ہیں؟
        """)
        
        st.markdown("---")
        st.markdown("**🔧 ماڈل کی معلومات / Model Info:**")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ ماڈل لوڈ ہو چکا ہے")
            model, stoi, itos, device = st.session_state.model_components
            st.text(f"آلات / Device: {device}")
            st.text(f"ذخیرہ الفاظ / Vocabulary: {len(itos)} الفاظ")
        else:
            st.warning("⏳ ماڈل لوڈ ہو رہا ہے")
        
        # Clear chat button
        if st.button("🗑️ بات چیت صاف کریں / Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Example prompts
        st.markdown("---")
        st.markdown("**💡 فوری سوالات / Quick Questions:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("سلام"):
                st.session_state.user_input = "سلام"
        with col2:
            if st.button("آپ کیسے ہیں؟"):
                st.session_state.user_input = "آپ کیسے ہیں؟"
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("شکریہ"):
                st.session_state.user_input = "شکریہ"
        with col4:
            if st.button("خدا حافظ"):
                st.session_state.user_input = "خدا حافظ"
    
    # Main chat area
    st.markdown("---")
    
    # Load model
    if not load_chatbot_model():
        st.error("ماڈل لوڈ نہیں ہو سکا۔ براہ کرم فائلوں کی جانچ پڑتال کریں۔")
        return
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div style="text-align: left; color: #1565c0; font-weight: bold;">👤 آپ / You:</div>
                    <div class="urdu-text">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div style="text-align: left; color: #7b1fa2; font-weight: bold;">🤖 بوٹ / Bot:</div>
                    <div class="urdu-text">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Initialize user_input in session state if not exists
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    # Use form to prevent reload on enter
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input(
            "اپنا پیغام یہاں لکھیں...",
            value=st.session_state.user_input,
            key="input_widget",
            placeholder="اردو میں لکھیں یا انگریزی میں ٹائپ کریں...",
            help="Enter your message in Urdu or English"
        )
        
        submit_button = st.form_submit_button(label="➡️ بھیجیں / Send")
    
    # Process user input
    if submit_button and user_input.strip():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        
        # Generate bot response
        with st.spinner("🤔 سوچ رہا ہوں..."):
            try:
                model, stoi, itos, device = st.session_state.model_components
                
                # Add small delay for better UX
                time.sleep(0.5)
                
                # Generate response
                response = generate_response(model, user_input.strip(), stoi, itos, device)
                
                # Add bot response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"معذرت، میں اس وقت جواب نہیں دے سکتا۔ مسئلہ: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear the input for next message
        st.session_state.user_input = ""
        st.rerun()

if __name__ == "__main__":
    main()
