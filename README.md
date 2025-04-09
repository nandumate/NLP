# NLP Assignments 

This repository contains multiple Natural Language Processing (NLP) assignments implemented using Python and popular NLP libraries like **NLTK**, **spaCy**, **scikit-learn**, **PyTorch**, and **Gensim**. Each assignment explores fundamental and advanced concepts in NLP, ranging from tokenization and stemming to transformer models and sentiment analysis.

---

## 📂 Assignment List

### ✅ Assignment 1: Tokenization & Stemming
- **Tokenization Techniques**:
  - Whitespace Tokenization
  - Punctuation-based Tokenization
  - Treebank Tokenizer (NLTK)
  - Tweet Tokenizer (NLTK)
  - MWE (Multi-Word Expression) Tokenizer
- **Stemming**:
  - Porter Stemmer
  - Snowball Stemmer
- **Lemmatization**:
  - Performed using `WordNetLemmatizer` from NLTK or `spaCy`


---

### ✅ Assignment 2: Bag-of-Words, TF-IDF, Word2Vec
- **Bag-of-Words**:
  - Count Occurrence
  - Normalized Count Occurrence
- **TF-IDF**:
  - Using `TfidfVectorizer` from `sklearn`
- **Word Embeddings**:
  - Generated using `Word2Vec` from `Gensim`


---

### ✅ Assignment 3: Text Preprocessing & Representation
- **Text Cleaning**:
  - Lowercasing, Removing Punctuation, Numbers, Special Characters
- **Lemmatization**:
  - Using `spaCy` or NLTK
- **Stop Word Removal**:
  - Using `nltk.corpus.stopwords` or `spaCy`
- **Label Encoding**:
  - Using `LabelEncoder` from `sklearn`
- **TF-IDF Vector Representation**
- **Output Saving**:
  - Cleaned data and TF-IDF vectors saved using `pickle` or `csv`


---

### ✅ Assignment 4: Transformer from Scratch using PyTorch
- Built a basic Transformer model inspired by Vaswani et al.'s paper *"Attention is All You Need"*.
- Includes:
  - Positional Encoding
  - Multi-Head Self Attention
  - Encoder & Decoder blocks
  - Masking techniques
- Implemented using PyTorch

📁 Files: `assignment_4_transformer_from_scratch.py`

---

### ✅ Assignment 5: Morphology Study
- Studied the **structure of words** using morphology.
- Created **Add/Delete tables** to understand prefixes, suffixes, and root words.
- Explored:
  - Inflectional and Derivational Morphology
  - Morphological Parsing

---

### ✅ Assignment 6: Advanced Sentiment Analysis
- Used pretrained deep learning models for sentiment classification:
  - BERT (via `transformers` library)
  - RoBERTa
- Fine-tuned on sample datasets (e.g., IMDb, SST)


---

### ✅ Assignment 7: Auto-Complete Using N-Gram Models
- Implemented **N-gram Language Models** (Unigram, Bigram, Trigram)
- Built a **predictive text system** for next-word suggestion
- Applications: Author Identification, Machine Translation, Speech Recognition


---

## 🛠️ Requirements

Install all dependencies using:
```bash
pip install -r requirements.txt
