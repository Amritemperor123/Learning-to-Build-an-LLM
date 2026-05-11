# Token Embedding Setup & Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install core packages:
```bash
pip install sentence-transformers torch transformers pandas pyarrow h5py numpy scikit-learn
```

### 2. Basic Usage

#### **Generate Sentence-Level Embeddings**
```python
from TokenEmbedding import TokenEmbedder

# Initialize embedder
embedder = TokenEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process text file
vectors, metadata, texts = embedder.process_file("your_data.txt", embedding_type="sentence")

# Save vectors (choose format)
embedder.save_vectors(vectors, metadata, "embeddings", format="parquet")
```

#### **Generate Token-Level Embeddings**
```python
# Token-level embeddings for granular control
vectors, metadata, texts = embedder.process_file("your_data.txt", embedding_type="token")

# Metadata includes token lists for reconstruction
print(metadata["token_lists"])  # [[token1, token2, ...], ...]
```

#### **Load Pre-computed Embeddings**
```python
# Load vectors and metadata
vectors, metadata = embedder.load_vectors("embeddings.parquet")

print(f"Shape: {vectors.shape}")
print(f"Model: {metadata['model']}")
```

---

## Core Features

### 🎯 **Three Embedding Types**

| Type | Best For | Example |
|------|----------|---------|
| **Sentence** | Document-level tasks, classification | Entire document as one vector |
| **Token** | Fine-grained analysis, sequence tasks | Each word/token as separate vector |
| **Custom** | Domain-specific needs | Mix of both with preprocessing |

### 💾 **Four Storage Formats**

| Format | Size | Speed | Use Case |
|--------|------|-------|----------|
| **Parquet** | Medium | Fast | ML pipelines, Pandas integration |
| **NumPy** | Large | Fastest | Quick experimentation, research |
| **HDF5** | Small (compressed) | Medium | Production, large-scale data |
| **Vector DB** | Medium | Slowest | Real-time search, semantic retrieval |

---

## Configuration Options

### **Embedder Parameters**
```python
embedder = TokenEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Pre-trained model
    embedding_dim=384,                                    # Output dimension (auto-detected)
    batch_size=64                                         # Batch size for GPU efficiency
)
```

### **Available Models** (from sentence-transformers)
- `all-MiniLM-L6-v2` - Fast, 384-dim (default, recommended for most cases)
- `all-mpnet-base-v2` - Better quality, 768-dim (slower)
- `paraphrase-mpnet-base-v2` - Semantic similarity, 768-dim
- `all-roberta-large-v1` - High quality, 1024-dim (slow)

Choose based on speed vs. quality trade-off.

---

## Common Workflows

### **Workflow 1: Train a Text Classification Model**
```python
from TokenEmbedding import TokenEmbedder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Generate embeddings
embedder = TokenEmbedder()
vectors, metadata, texts = embedder.process_file("training_data.txt")

# 2. Prepare labels (example: first N are class 0, rest are class 1)
labels = [0] * (len(vectors) // 2) + [1] * (len(vectors) - len(vectors) // 2)

# 3. Train classifier
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Evaluate
print(f"Accuracy: {model.score(X_test, y_test)}")
```

### **Workflow 2: Find Similar Documents**
```python
from sklearn.metrics.pairwise import cosine_similarity

vectors, metadata, texts = embedder.process_file("docs.txt")

# Find top-5 similar documents to query
query = "machine learning is important"
query_vector = embedder.embed_sentences([query])[0]

similarities = cosine_similarity([query_vector], vectors)[0]
top_indices = similarities.argsort()[-5:][::-1]

for idx in top_indices:
    print(f"[{similarities[idx]:.4f}] {texts[idx]}")
```

### **Workflow 3: Semantic Search with Vector Database**
```python
# Install: pip install pinecone-client
import pinecone

# Initialize
pinecone.init(api_key="your_key", environment="us-west1-gcp")
index = pinecone.Index("embeddings")

# Upload vectors
vectors, metadata, texts = embedder.process_file("docs.txt")
for i, vec in enumerate(vectors):
    index.upsert([(str(i), vec, {"text": texts[i]})])

# Query
results = index.query(query_vector=query_vec, top_k=5)
```

---

## Performance Tips

### 🚀 **Speed Optimization**
```python
# 1. Use smaller model for speed
embedder = TokenEmbedder(model_name="all-MiniLM-L6-v2")  # Fast

# 2. Increase batch size (if GPU memory allows)
embedder.batch_size = 128  # Default is 64

# 3. Use GPU (automatic with PyTorch)
# Ensure torch has CUDA: torch.cuda.is_available() returns True
```

### 💾 **Storage Optimization**
```python
# 1. Use HDF5 with compression
embedder.save_vectors(vectors, metadata, "embeddings", format="hdf5")

# 2. Use float32 instead of float64
vectors = vectors.astype(np.float32)

# 3. Quantize vectors (128-bit to 8-bit)
# Advanced: Use vector compression libraries
```

### 🧠 **Memory Optimization**
```python
# Process large files in chunks
batch_size = 10000
all_vectors = []
all_texts = []

with open("large_file.txt") as f:
    lines = f.readlines()

for i in range(0, len(lines), batch_size):
    batch = [l.strip() for l in lines[i:i+batch_size]]
    batch_vectors = embedder.embed_sentences(batch)
    all_vectors.append(batch_vectors)
    all_texts.extend(batch)

full_vectors = np.vstack(all_vectors)
```

---

## Troubleshooting

### **Problem: "No module named 'sentence_transformers'"**
**Solution:**
```bash
pip install sentence-transformers
```

### **Problem: "CUDA out of memory"**
**Solution:**
```python
# Reduce batch size
embedder.batch_size = 16

# Use CPU instead
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### **Problem: "File not found" error**
**Solution:**
- Ensure the input file exists in the correct path
- Use absolute paths: `os.path.abspath("file.txt")`

### **Problem: Slow embeddings generation**
**Solution:**
1. Check if GPU is being used: `torch.cuda.is_available()`
2. Install PyTorch with CUDA support
3. Use a faster model: `all-MiniLM-L6-v2`
4. Increase batch size (if memory allows)

---

## Project Files

```
Stage1/
├── TokenEmbedding.py                 # Main embedder class
├── Tokenizer.py                      # Tokenization utilities
├── Vocabulary.py                     # Vocabulary management
├── BytePairEncoding.py              # BPE implementation
├── requirements.txt                  # Dependencies
├── examples.py                       # Usage examples
├── VECTOR_STORAGE_STRATEGY.md       # Storage recommendations (THIS FILE)
└── README_SETUP.md                  # This setup guide

embeddings/ (created after processing)
├── embeddings.parquet               # Parquet format
├── embeddings_metadata.json         # Metadata
├── embeddings.npy                   # NumPy format
└── embeddings.h5                    # HDF5 format
```

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run examples: `python examples.py`
3. ✅ Try with your data: `embedder.process_file("your_file.txt")`
4. ✅ Choose storage format from VECTOR_STORAGE_STRATEGY.md
5. ✅ Integrate with your model training pipeline

---

## Additional Resources

- **Sentence Transformers Docs:** https://www.sbert.net/
- **HuggingFace Models:** https://huggingface.co/sentence-transformers
- **Vector Database Options:**
  - Pinecone: https://www.pinecone.io/
  - Weaviate: https://weaviate.io/
  - Milvus: https://milvus.io/
  - Chroma: https://www.trychroma.com/

---

**Questions?** Check the inline code comments or refer to the examples.py file for working code samples!
