#!/bin/bash
# setup.sh - Streamlit Cloud deployment setup

apt-get update
apt-get install -y fonts-noto fonts-noto-cjk fonts-noto-color-emoji
mkdir -p /home/appuser/.fonts

# Fix for TensorFlow compatibility
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install specific versions to avoid conflicts
pip install protobuf==3.20.3
pip install typing-extensions==4.5.0

# Clear cache to avoid space issues
pip cache purge
