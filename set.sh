#!/bin/bash
# setup.sh - Streamlit Cloud deployment setup

apt-get update
apt-get install -y fonts-noto fonts-noto-cjk fonts-noto-color-emoji
mkdir -p /home/appuser/.fonts

# Fix for TensorFlow compatibility
pip install --upgrade pip
pip install protobuf==3.20.3

# Install Urdu fonts if needed
curl -L -o /home/appuser/.fonts/NotoNastaliqUrdu-Regular.ttf "https://github.com/notofonts/noto-fonts/raw/main/unhinted/variable-ttf/NotoNastaliqUrdu%5Bwght%5D.ttf"
fc-cache -f -v