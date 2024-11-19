# Transformer Translation Model - From Scratch in PyTorch  

![Transformer Architecture](Transformer_model_arch.png)  

This repository contains a PyTorch implementation of the Transformer model built from scratch for machine translation tasks. Inspired by the seminal paper *"Attention Is All You Need"*, this project offers a hands-on way to understand and experiment with the Transformer architecture and related concepts like self-attention and encoder-decoder mechanisms.

---

## 📂 Repository Contents  

- **`Attention-Is-All-You-Need_paper.pdf`**  
  A copy of the original research paper that introduced the Transformer architecture. This paper is an excellent reference for understanding the theoretical foundations of this implementation.

- **`Beam_Search.ipynb`**  
  A Jupyter Notebook demonstrating the Beam Search decoding method for predicting translations. It provides a comparison with greedy decoding for better insights into translation quality improvements.

- **`config.py`**  
  This file contains all configurable settings for the model, including hyperparameters like `epochs`, `batch_size`, `optimizer`, and other training parameters.

- **`dataset.py`**  
  This script includes the code for a custom tokenizer built from scratch to preprocess text data for the model.

- **`model.py`**  
  The core implementation of the Transformer model, containing all components such as:  
  - Input Embedding  
  - Positional Encoding  
  - Add & Normalize  
  - Self-Attention  
  - Multi-Head Attention Block  
  - Encoder and Decoder Blocks  

- **`requirements.txt`**  
  A list of required Python libraries for running the project. Install them using:  
  ```bash
  pip install -r requirements.txt

🚀 **Features**
Implementation of the Transformer architecture from scratch using PyTorch.
Tokenization and preprocessing pipeline built in-house.
Configurable hyperparameters for flexible experimentation.
Support for both Greedy and Beam Search decoding strategies.
Training and testing scripts to facilitate end-to-end experimentation.
Clear modular structure for easy extension and understanding.

📜 **Getting Started**
Clone this repository:

bash
git clone https://github.com/username/transformer-pytorch-translation.git
cd transformer-pytorch-translation


Install the dependencies:

for requirements
pip install -r requirements.txt


**Author**
Karthikeya Valmiki

