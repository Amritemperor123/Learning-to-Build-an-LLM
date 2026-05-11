"""
Example: Token Embedding Pipeline
Demonstrates various use cases for the TokenEmbedder class.
"""

import numpy as np
from TokenEmbedding import TokenEmbedder
import json

def example_1_basic_sentence_embeddings():
    """Example 1: Basic sentence-level embeddings with Parquet storage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Sentence Embeddings")
    print("="*60)
    
    embedder = TokenEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create sample data
    sample_text = """
Natural language processing is fascinating.
Deep learning has revolutionized AI.
Embeddings capture semantic meaning.
Vector databases enable efficient search.
    """.strip()
    
    with open("sample_data.txt", "w") as f:
        for line in sample_text.split("\n"):
            if line.strip():
                f.write(line + "\n")
    
    # Process and save
    vectors, metadata, texts = embedder.process_file("sample_data.txt", embedding_type="sentence")
    
    print(f"\n📊 Summary:")
    print(f"   Documents: {vectors.shape[0]}")
    print(f"   Embedding Dimension: {vectors.shape[1]}")
    print(f"   Metadata: {json.dumps(metadata, indent=2)}")
    
    # Save in multiple formats
    embedder.save_vectors(vectors, metadata, "embeddings_example1", format="parquet")
    embedder.save_vectors(vectors, metadata, "embeddings_example1", format="numpy")


def example_2_token_level_embeddings():
    """Example 2: Token-level embeddings for granular control."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Token-Level Embeddings")
    print("="*60)
    
    embedder = TokenEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectors, metadata, texts = embedder.process_file("sample_data.txt", embedding_type="token")
    
    print(f"\n📊 Summary:")
    print(f"   Total Tokens: {vectors.shape[0]}")
    print(f"   Unique Tokens: {metadata['n_unique_tokens']}")
    print(f"   Embedding Dimension: {vectors.shape[1]}")
    print(f"   Token Lists (first doc): {metadata['token_lists'][0]}")
    
    embedder.save_vectors(vectors, metadata, "embeddings_tokens", format="hdf5")


def example_3_loading_vectors():
    """Example 3: Load embeddings and use them."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Loading and Using Embeddings")
    print("="*60)
    
    embedder = TokenEmbedder()
    
    # Load from different formats
    print("\n1️⃣  Loading from Parquet:")
    vectors_parquet, meta_parquet = embedder.load_vectors("embeddings_example1.parquet")
    
    print("\n2️⃣  Loading from NumPy:")
    vectors_numpy, meta_numpy = embedder.load_vectors("embeddings_example1.npy")
    
    print("\n3️⃣  Verifying they're identical:")
    print(f"   Arrays equal: {np.allclose(vectors_parquet, vectors_numpy)}")


def example_4_similarity_search():
    """Example 4: Find similar documents using embeddings."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Similarity Search")
    print("="*60)
    
    embedder = TokenEmbedder()
    vectors, metadata, texts = embedder.process_file("sample_data.txt")
    
    # Query: find similar documents to the first one
    query_vector = vectors[0]
    
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_vector], vectors)[0]
    
    print(f"\n🔍 Query: '{texts[0]}'")
    print(f"\nSimilarity scores to all documents:")
    
    for idx, (text, score) in enumerate(sorted(zip(texts, similarities), 
                                               key=lambda x: x[1], reverse=True)):
        print(f"   [{score:.4f}] {text[:50]}...")


def example_5_batch_processing():
    """Example 5: Batch processing for large files."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing Large Files")
    print("="*60)
    
    embedder = TokenEmbedder(batch_size=32)
    
    # Create larger sample dataset
    large_text = "\n".join([
        f"This is sample document number {i} with some text content."
        for i in range(100)
    ])
    
    with open("large_data.txt", "w") as f:
        f.write(large_text)
    
    vectors, metadata, texts = embedder.process_file("large_data.txt")
    
    print(f"\n📊 Processed {vectors.shape[0]} documents in batches of {embedder.batch_size}")
    print(f"   Shape: {vectors.shape}")
    
    # Save with compression
    embedder.save_vectors(vectors, metadata, "embeddings_large", format="hdf5")


def example_6_pytorch_integration():
    """Example 6: Integration with PyTorch (if available)."""
    print("\n" + "="*60)
    print("EXAMPLE 6: PyTorch Integration")
    print("="*60)
    
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        import pandas as pd
        
        # Load embeddings
        df = pd.read_parquet("embeddings_example1.parquet")
        vectors = df.values
        
        # Create PyTorch dataset
        tensor_vectors = torch.FloatTensor(vectors)
        dataset = TensorDataset(tensor_vectors)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print(f"\n🔥 PyTorch DataLoader created:")
        print(f"   Batch size: 2")
        print(f"   Total batches: {len(dataloader)}")
        
        for batch_idx, (batch_vectors,) in enumerate(dataloader):
            print(f"\n   Batch {batch_idx + 1}: shape {batch_vectors.shape}")
            if batch_idx == 0:  # Show first batch details
                print(f"   First vector sample: {batch_vectors[0][:5].numpy()}...")
        
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping this example.")
        print("   Install with: pip install torch")


def example_7_metadata_inspection():
    """Example 7: Inspect and work with metadata."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Metadata Inspection")
    print("="*60)
    
    embedder = TokenEmbedder()
    vectors, metadata, texts = embedder.process_file("sample_data.txt")
    
    print(f"\n📋 Full Metadata:")
    print(json.dumps(metadata, indent=2, default=str))
    
    print(f"\n✨ Key Information:")
    print(f"   - Model used: {metadata['model']}")
    print(f"   - Embedding dimension: {metadata['embedding_dim']}")
    print(f"   - Type: {metadata['type']}")
    print(f"   - Number of documents: {metadata['n_documents']}")


if __name__ == "__main__":
    print("🚀 Token Embedding Examples Pipeline")
    
    # Run examples
    example_1_basic_sentence_embeddings()
    example_2_token_level_embeddings()
    example_3_loading_vectors()
    example_4_similarity_search()
    example_5_batch_processing()
    example_6_pytorch_integration()
    example_7_metadata_inspection()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)
    print("\n📁 Generated files:")
    print("   - embeddings_example1.parquet / .npy")
    print("   - embeddings_tokens.h5")
    print("   - embeddings_large.h5")
    print("   - Various _metadata.json files")
    print("\n💡 Check VECTOR_STORAGE_STRATEGY.md for detailed recommendations!")
