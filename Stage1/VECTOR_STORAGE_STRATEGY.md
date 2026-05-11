# Vector Storage Strategy Guide

## Overview
After generating embeddings, choosing the right storage format is crucial for model training efficiency, scalability, and ease of use. This guide provides recommendations for different use cases.

---

## 1. Storage Format Comparison

| Format | Use Case | Pros | Cons | Speed | Space |
|--------|----------|------|------|-------|-------|
| **Parquet** | ML pipelines, Pandas integration | Columnar, integrates with ML tools | Requires Pandas/Arrow | ⚡⚡⚡ | 📦 |
| **NumPy (.npy)** | Quick experimentation | Fast I/O, simple, universal | No compression, not columnar | ⚡⚡⚡⚡ | 📦📦 |
| **HDF5** | Large-scale production | Compression, hierarchical, streaming | Steeper learning curve | ⚡⚡ | 📦 |
| **Vector DBs** | Real-time retrieval, semantic search | Indexing, fast similarity search | Added complexity | ⚡ | 📦 |

---

## 2. Recommended Approaches by Scenario

### **Scenario A: Training a Small to Medium Model (< 1 GB)**
**Recommended: Parquet Format**
```python
# Save
embedder.save_vectors(vectors, metadata, "embeddings", format="parquet")

# Load and use with PyTorch
import pandas as pd
df = pd.read_parquet("embeddings.parquet")
vectors = df.values  # Convert to numpy array

# Create PyTorch dataset
from torch.utils.data import TensorDataset, DataLoader
import torch
dataset = TensorDataset(torch.FloatTensor(vectors))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
**Why:** Seamless integration with ML tools (scikit-learn, PyTorch Lightning, HuggingFace).

---

### **Scenario B: Large-Scale Production (> 1 GB)**
**Recommended: HDF5 Format**
```python
# Save with compression
embedder.save_vectors(vectors, metadata, "embeddings", format="hdf5")

# Load in chunks (memory-efficient)
import h5py
with h5py.File("embeddings.h5", "r") as f:
    embeddings = f["embeddings"]  # Don't load all at once
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        # Process batch
```
**Why:** Compression reduces storage 30-50%, supports streaming large files without loading everything into memory.

---

### **Scenario C: Real-Time Semantic Search / Retrieval**
**Recommended: Vector Database**
```python
# Install: pip install pinecone-client  (or Weaviate, Milvus, Chroma)
import pinecone

# Initialize
pinecone.init(api_key="your_api_key", environment="us-west1-gcp")
index = pinecone.Index("embeddings-index")

# Upload vectors
vectors_to_upsert = [(str(i), embedding, {"doc_id": i}) 
                     for i, embedding in enumerate(vectors)]
index.upsert(vectors=vectors_to_upsert)

# Query - find similar documents
results = index.query(query_vector=new_embedding, top_k=10)
```
**Why:** Built-in similarity search, scalable to billions of vectors, low-latency retrieval.

---

### **Scenario D: Quick Experimentation / Prototyping**
**Recommended: NumPy Format**
```python
# Save
embedder.save_vectors(vectors, metadata, "embeddings", format="numpy")

# Load
vectors = np.load("embeddings.npy")
```
**Why:** Fastest I/O, lightweight, perfect for iterative development.

---

## 3. Multi-Format Storage Strategy (Recommended)

For maximum flexibility, store in **multiple formats**:

```python
vectors, metadata, texts = embedder.process_file("data.txt")

# Store in all formats for different pipelines
embedder.save_vectors(vectors, metadata, "embeddings", format="parquet")  # For ML pipelines
embedder.save_vectors(vectors, metadata, "embeddings", format="numpy")    # For quick access
embedder.save_vectors(vectors, metadata, "embeddings", format="hdf5")     # For production
```

### Directory Structure:
```
embeddings/
├── embeddings.parquet          # ML pipeline
├── embeddings_metadata.json    # Metadata
├── embeddings.npy              # NumPy format
├── embeddings.h5               # HDF5 format
└── vectors.db                  # Vector database (optional)
```

---

## 4. Metadata Storage Best Practices

Always save metadata alongside vectors:

```json
{
  "type": "sentence",
  "n_documents": 1000,
  "embedding_dim": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "created_at": "2026-05-11T10:30:00",
  "source_file": "data.txt",
  "preprocessing_steps": ["lowercase", "remove_special_chars"],
  "token_lists": [[...], [...], ...]  # For token-level embeddings
}
```

**Why:** Track data lineage, model version, and reproduce embeddings.

---

## 5. Optimization Tips

### **Compression**
```python
# HDF5 with compression (50% space reduction)
import h5py
with h5py.File("embeddings.h5", "w") as f:
    f.create_dataset("embeddings", data=vectors, compression="gzip", compression_opts=4)
```

### **Quantization** (Reduce precision for smaller files)
```python
# Store as float32 instead of float64
vectors_fp32 = vectors.astype(np.float32)
np.save("embeddings_fp32.npy", vectors_fp32)
```

### **Batching for Large Files**
```python
# Process and save incrementally
batch_size = 10000
for i in range(0, len(texts), batch_size):
    batch_vectors = embedder.embed_sentences(texts[i:i+batch_size])
    # Save or append to database
```

---

## 6. Production Checklist

- ✅ Store metadata with vectors
- ✅ Use versioning for model and preprocessing
- ✅ Test loading/saving round-trip
- ✅ Document storage location and access pattern
- ✅ Set up backup strategy
- ✅ Monitor storage growth and cleanup old versions
- ✅ Use compression for large datasets
- ✅ Consider vector databases for retrieval-heavy workflows

---

## 7. Integration with Training Pipeline

### Example: PyTorch Lightning Training
```python
from torch.utils.data import DataLoader, Dataset
import torch

class EmbeddingDataset(Dataset):
    def __init__(self, vectors_path, metadata_path):
        if vectors_path.endswith('.parquet'):
            import pandas as pd
            self.vectors = pd.read_parquet(vectors_path).values
        elif vectors_path.endswith('.npy'):
            self.vectors = np.load(vectors_path)
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.vectors[idx])

# Usage
dataset = EmbeddingDataset("embeddings.parquet", "embeddings_metadata.json")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Summary

| Scenario | Format | Command |
|----------|--------|---------|
| ML Pipelines | Parquet | `embedder.save_vectors(..., format="parquet")` |
| Production Scale | HDF5 | `embedder.save_vectors(..., format="hdf5")` |
| Experimentation | NumPy | `embedder.save_vectors(..., format="numpy")` |
| Semantic Search | Vector DB | Pinecone / Weaviate / Milvus |
| All Scenarios | Multi-format | Save in all 3 formats |

Choose based on your pipeline's requirements for I/O speed, storage efficiency, and use-case complexity.
