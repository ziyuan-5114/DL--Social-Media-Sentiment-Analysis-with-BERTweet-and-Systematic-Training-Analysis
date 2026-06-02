# Social Media Sentiment Analysis with BERTweet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 📌 Overview

This project builds a **three-class (positive/neutral/negative) sentiment analysis system** for tweets using **BERTweet** – a pre-trained language model specifically designed for social media. We systematically compare different loss functions (Focal Loss vs. Label Smoothing Cross-Entropy), learning rates (1e-5 ~ 1e-4), and batch sizes (8 ~ 128). Experimental results show that the combination of **BERTweet + Focal Loss** achieves the best test accuracy of **80.00%** and macro-F1 of **0.8034**.

## 🗂️ Dataset

- **Source**: [Kaggle Social Media Sentiment Analysis](https://www.kaggle.com/datasets/mdismiellhossenabir/sentiment-analysis)
- **Total samples**: 4,727 (balanced across three classes, ~33% each)
- **Split**: 70% train (3,309), 15% validation (709), 15% test (709)
- **Average text length**: 13.2 words, 95th percentile ≤ 25 words → `max_length=128`

### Preprocessing (light strategy)

- Remove URLs, replace `@username` with `<USER>`
- Keep hashtag text (remove `#` only)
- Convert emojis to text descriptions (fallback to Unicode)
- Lowercase, remove extra spaces  
  *(punctuation and typos are kept – BERTweet handles them natively)*

## 🧠 Model Architecture

Input Tweet → BERTweet (frozen or fine-tuned) → [CLS] token → Dropout(0.3) → Linear(768→3) → Softmax


- **Backbone**: `BERTweet` (135M parameters) – pre-trained on 850M English tweets
- **Classification head**: lightweight single linear layer
- **Loss functions**: Focal Loss (main) vs Label Smoothing Cross-Entropy (baseline)
- **Optimizer**: AdamW with cosine annealing + warmup

## ⚙️ Hyperparameter Tuning

We performed systematic sweeps and selected the optimal configuration:

| Hyperparameter | Tested values           | Best choice |
|----------------|-------------------------|--------------|
| Learning rate  | 1e-5, 2e-5, 5e-5, 1e-4 | **1e-4**     |
| Batch size     | 8, 16, 32, 64, 128      | **32**       |
| Loss function  | Focal Loss / LabelSmoothingCE | **Focal Loss** |
| Epochs         | 10                      | (early stop on val macro-F1) |


## 🔍 Error Analysis & Limitations

Dataset size: Only 4.7k samples → potential overfitting (validation plateau after 7-8 epochs)

Neutral class ambiguity: Linguistic similarity with weak sentiment expressions

Metadata unused: Timestamps and platform info (Twitter/Facebook/Instagram) were discarded – may contain sentiment patterns

Complex phenomena: Sarcasm, irony, and emoji combinations not explicitly modeled

Inference cost: BERTweet (135M) is heavy for CPU or real-time deployment

## 🔮 Future Work

Data augmentation (back-translation, synonym replacement)

Fine-grained neutral modeling (sentiment intensity regression, multi-task learning)

Fuse metadata (time, platform) with text via attention

Sarcasm/irony detection modules or larger models (BERTweet-large)

Model compression (knowledge distillation, quantization) for edge deployment

Online learning (LoRA) to adapt to evolving tweet language
