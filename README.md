# N-Gram Language Model: Africa Galore Dataset

A Python implementation of n-gram language models for text generation using the Africa Galore dataset. This project demonstrates fundamental concepts in natural language processing, including tokenization, n-gram extraction, probability estimation, and text generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

## 🔍 Overview

This project implements an n-gram language model from scratch to understand how language models capture patterns in text and generate new content. The model learns statistical patterns from the Africa Galore dataset and uses conditional probabilities to predict and generate text sequences.

## ✨ Features

- **Tokenization**: Space-based tokenizer for splitting text into tokens
- **N-gram Generation**: Flexible n-gram extraction (unigrams, bigrams, trigrams, etc.)
- **Probability Estimation**: Computes conditional probabilities P(B|A) from n-gram counts
- **Text Generation**: Generates coherent text continuations based on learned probabilities
- **Multiple Model Orders**: Support for bigram, trigram, and higher-order n-gram models

## 📚 Dataset

The **Africa Galore** dataset consists of 232 synthetically generated paragraphs focusing on African culture, history, and geography. The dataset was created using Google's Gemini language model and covers topics including:

- African music genres (Afrobeat, Highlife, Juju, Mbalax, Soukous, Isicathamiya)
- Traditional textiles (Kente cloth, Bogolanfini, Adire, Kanga)
- Cultural practices and heritage
- Geographic landmarks and locations

Dataset source: [Africa Galore JSON](https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Google Colab (recommended) or local Python environment

### Required Libraries
```bash
pip install pandas
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/ngram-language-model.git
cd ngram-language-model
```

## 💻 Usage

### 1. Basic Setup
```python
import random
from collections import Counter, defaultdict
import pandas as pd

# Load the dataset
africa_galore = pd.read_json(
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"
)
dataset = africa_galore["description"]
```

### 2. Generate N-grams
```python
def generate_ngrams(text: str, n: int) -> list[tuple[str]]:
    tokens = space_tokenize(text)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Example
text = "Table Mountain is tall."
bigrams = generate_ngrams(text, n=2)
# Output: [('Table', 'Mountain'), ('Mountain', 'is'), ('is', 'tall.')]
```

### 3. Build N-gram Model
```python
# Build a trigram model
trigram_model = build_ngram_model(dataset, n=3)

# Example probability distribution
context = "looking for"
print(trigram_model[context])
```

### 4. Generate Text
```python
prompt = "Jide was hungry so she went looking for"
generated_text = generate_next_n_tokens(
    n=3,
    ngram_model=trigram_model,
    prompt=prompt,
    num_tokens_to_generate=10
)
print(generated_text)
```

## 🔧 Implementation Details

### Core Functions

#### 1. `space_tokenize(text: str) -> list[str]`
Splits text into tokens based on spaces.

#### 2. `generate_ngrams(text: str, n: int) -> list[tuple[str]]`
Generates n-grams of length n from tokenized text.

#### 3. `get_ngram_counts(dataset: list[str], n: int) -> dict[str, Counter]`
Computes n-gram counts where:
- Keys: (n-1)-token contexts
- Values: Counter objects with next token frequencies

#### 4. `build_ngram_model(dataset: list[str], n: int) -> dict[str, dict[str, float]]`
Builds probability distributions:
- **Formula**: P(B|A) = Count(A B) / Count(A)
- Returns nested dictionary of conditional probabilities

#### 5. `generate_next_n_tokens(...)`
Iteratively generates tokens using the n-gram model and random sampling.

## 📊 Key Findings

### Most Common N-grams

**Bigrams:**
- ("is", "a"): 144 occurrences
- ("of", "the"): 100 occurrences
- ("and", "the"): 69 occurrences

**Trigrams:**
- ("went", "looking", "for"): 32 occurrences
- ("a", "symbol", "of"): 18 occurrences
- ("was", "hungry", "so"): 18 occurrences

### Data Sparsity

- **Bigrams**: 99.95% of possible combinations have zero counts
- **Trigrams**: 99.98% of possible combinations have zero counts
- Higher-order n-grams suffer from exponentially increasing sparsity

## ⚠️ Limitations

1. **Data Sparsity**: Many n-gram combinations never appear in the dataset, leading to zero probabilities
2. **Context Length**: Cannot generate continuations for unseen contexts
3. **Long-range Dependencies**: Limited ability to capture dependencies beyond n-1 tokens
4. **Exponential Growth**: Number of possible n-grams grows exponentially with n (5^n for vocabulary of 5)
5. **Generation Failures**: Model fails when encountering contexts not present in training data

## 🔮 Future Improvements

- Implement smoothing techniques (Laplace, Good-Turing)
- Add back-off strategies for unseen n-grams
- Explore higher-order n-gram models with larger datasets
- Compare with neural language models (RNNs, Transformers)
- Add evaluation metrics (perplexity, BLEU scores)

## 📖 References

[1] Ronen Eldan and Yuanzhi Li. 2023. Tiny Stories: How Small Can Language Models Be and Still Speak Coherent English. arXiv:2305.07759. Retrieved from https://arxiv.org/pdf/2305.07759.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Google DeepMind AI Foundations Course
- Africa Galore dataset created using Google's Gemini
- Inspired by the TinyStories project

---

**Note**: This project is for educational purposes and demonstrates fundamental concepts in natural language processing and language modeling.
