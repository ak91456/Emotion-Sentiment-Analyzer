# Emotion Sentiment Analyzer 🧠💬

[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains three deep learning models — **RNN**, **LSTM**, and **GRU** — implemented in Jupyter notebooks to perform **emotion-based sentiment analysis** on text data. The models aim to classify text samples (like tweets or messages) into different emotion categories such as **joy, sadness, anger, fear**, etc.

---

## 📁 Notebooks Included

- `Rnn model.ipynb`  
  → Implements a basic Recurrent Neural Network (RNN) to classify emotional sentiments.

- `LSTM.ipynb`  
  → Uses Long Short-Term Memory (LSTM) layers for handling longer-term dependencies in text.

- `Gru_model.ipynb`  
  → Uses Gated Recurrent Units (GRU), a more efficient variant of LSTM with similar performance.

---

## 📊 Dataset

These models are built for use with emotion-labeled datasets. Each sample in the dataset includes:

- A **text sentence**
- A **label** indicating the emotion (e.g., happy, sad, angry)

> Suggested Dataset: [Hugging Face - Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)

Example:

| Text                           | Emotion  |
|--------------------------------|----------|
| "I'm feeling awesome today!"   | joy      |
| "Why did this have to happen?" | sadness  |
| "I'm so mad right now!"        | anger    |

---

## 🛠 Features

- Preprocessing (tokenization, padding)
- Embedding layer using Keras
- RNN/LSTM/GRU-based architectures
- Evaluation using accuracy and confusion matrix
- Training vs Validation loss/accuracy plots
- Model comparison summary

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- Jupyter Notebook / JupyterLab
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### 🧩 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/emotion-sentiment-analyzer.git
cd emotion-sentiment-analyzer
pip install -r requirements.txt
