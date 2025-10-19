# inference.py
import torch
import torch.nn as nn
import json
import re
import math

class UrduChatbotInference:
    def __init__(self, model_path="best_urdu_chatbot.pth", tokenizer_path="urdu_tokenizer.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 64
        
        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            self.tokenizer_config = json.load(f)
        
        self.word_index = self.tokenizer_config['word_index']
        self.reverse_word_index = {v: k for k, v in self.word_index.items()}
        self.vocab_size = self.tokenizer_config['vocab_size']
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load the trained model architecture"""
        # Import the model architecture from app.py
        from app import TransformerModel
        
        model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=256,
            enc_layers=2,
            dec_layers=2,
            n_heads=2,
            d_ff=512
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
        text = text.replace('\u0640', '')
        text = re.sub('[\u0622\u0623\u0625]', 'ا', text)
        text = re.sub('[\u064A\u06D0]', 'ی', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def text_to_sequence(self, text):
        """Convert text to sequence"""
        text = self.preprocess_text(text)
        tokens = re.findall(r'[\u0600-\u06FF]+|[.,?!؛؟]', text)
        sequence = [self.word_index.get('<sos>', 2)]
        
        for token in tokens:
            sequence.append(self.word_index.get(token, self.word_index.get('<OOV>', 1)))
        
        sequence.append(self.word_index.get('<eos>', 3))
        
        # Pad or truncate
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
            sequence[-1] = self.word_index.get('<eos>', 3)
        else:
            sequence.extend([0] * (self.max_len - len(sequence)))
        
        return torch.LongTensor([sequence])
    
    def sequence_to_text(self, sequence):
        """Convert sequence back to text"""
        tokens = []
        for token_id in sequence:
            if token_id == 0:  # PAD
                continue
            if token_id == self.word_index.get('<sos>', 2):
                continue
            if token_id == self.word_index.get('<eos>', 3):
                break
            token = self.reverse_word_index.get(token_id, '<UNK>')
            tokens.append(token)
        return ' '.join(tokens)
    
    def generate_response(self, input_text, decoding_strategy='greedy'):
        """Generate response for input text"""
        try:
            with torch.no_grad():
                input_sequence = self.text_to_sequence(input_text).to(self.device)
                
                if decoding_strategy == 'greedy':
                    return self.greedy_decode(input_sequence)
                else:
                    return self.beam_search_decode(input_sequence)
                    
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def greedy_decode(self, input_sequence):
        """Greedy decoding for inference"""
        enc_out = self.model.enc(input_sequence, self.model.make_src_mask(input_sequence))
        
        batch_size = input_sequence.size(0)
        ys = torch.full((batch_size, 1), self.word_index.get('<sos>', 2), 
                       dtype=torch.long, device=self.device)
        
        for _ in range(self.max_len - 1):
            tgt_mask = self.model.make_tgt_mask(ys)
            out = self.model.dec(ys, enc_out, tgt_mask=tgt_mask, 
                               memory_mask=self.model.make_src_mask(input_sequence))
            next_logits = out[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            
            if next_token.item() == self.word_index.get('<eos>', 3):
                break
        
        return self.sequence_to_text(ys[0].cpu().numpy())
    
    def beam_search_decode(self, input_sequence, beam_width=3):
        """Beam search decoding (simplified implementation)"""
        # For now, fallback to greedy
        return self.greedy_decode(input_sequence)