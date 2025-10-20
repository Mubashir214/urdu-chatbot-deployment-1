#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt', quiet=True)" || true

# Create necessary directories
mkdir -p .streamlit

echo "✅ Urdu Chatbot setup completed successfully!"
