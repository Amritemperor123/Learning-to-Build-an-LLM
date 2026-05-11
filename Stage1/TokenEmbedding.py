import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import re

class TokenEmbedder:
    """
    Token Embedder for converting text into dense vector representations.
    Supports both sentence-level and token-level embeddings with multiple storage formats.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 embedding_dim: int = None,
                 batch_size: int = 64):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the pre-trained embedding model (default: all-MiniLM-L6-v2).
            embedding_dim: Optional dimension override (projection layer, rarely needed).
            batch_size: Batch size for processing large texts (default: 64).
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        
        if embedding_dim is not None and embedding_dim != self.embedding_dim:
            print(f"⚠️  Model native dim ({self.embedding_dim}) differs from requested ({embedding_dim}).")

    def load_text(self, file_path: str) -> List[str]:
        """
        Reads text from file. Handles empty lines gracefully.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List of non-empty text lines.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✓ Loaded {len(texts)} lines from {file_path}")
        return texts

    def embed_sentences(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate sentence-level embeddings (entire text as one vector).
        
        Args:
            texts: List of text strings to embed.
            show_progress: Display progress information.
            
        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        if show_progress:
            print(f"📊 Generating sentence embeddings for {len(texts)} documents...")
        
        embeddings = self.model.encode(texts, batch_size=self.batch_size, 
                                       convert_to_numpy=True, show_progress_bar=show_progress)
        return embeddings

    def embed_tokens(self, texts: List[str], show_progress: bool = True) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Generate token-level embeddings by splitting texts into tokens.
        
        Args:
            texts: List of text strings to embed.
            show_progress: Display progress information.
            
        Returns:
            Tuple of (embeddings array, token_lists).
            embeddings shape: (total_tokens, embedding_dim)
            token_lists: List of token lists per document.
        """
        if show_progress:
            print(f"🔤 Generating token-level embeddings...")
        
        # Tokenize all texts
        all_tokens = []
        token_lists = []
        
        for text in texts:
            # Simple word tokenization (customize as needed)
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            token_lists.append(tokens)
            all_tokens.extend(tokens)
        
        # Remove duplicates for embedding (reduce compute)
        unique_tokens = list(set(all_tokens))
        
        if show_progress:
            print(f"   Found {len(unique_tokens)} unique tokens")
        
        # Embed unique tokens
        token_embeddings_dict = {}
        batch_size = self.batch_size
        
        for i in range(0, len(unique_tokens), batch_size):
            batch = unique_tokens[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_numpy=True, 
                                          show_progress_bar=False)
            for token, embedding in zip(batch, embeddings):
                token_embeddings_dict[token] = embedding
        
        # Reconstruct full token embeddings in original order
        full_embeddings = np.array([token_embeddings_dict[token] for token in all_tokens])
        
        return full_embeddings, token_lists

    def process_file(self, file_path: str, embedding_type: str = "sentence"):
        """
        Main pipeline: Load file → Generate embeddings → Return vectors.
        
        Args:
            file_path: Path to input text file.
            embedding_type: "sentence" or "token" level embeddings.
            
        Returns:
            Tuple of (vectors, metadata_dict).
        """
        print(f"\n📂 Processing {file_path}...")
        texts = self.load_text(file_path)
        
        if len(texts) == 0:
            raise ValueError("No text found in the input file.")
        
        if embedding_type == "sentence":
            vectors = self.embed_sentences(texts)
            metadata = {
                "type": "sentence",
                "n_documents": len(texts),
                "embedding_dim": int(self.embedding_dim),
                "model": self.model_name
            }
        elif embedding_type == "token":
            vectors, token_lists = self.embed_tokens(texts)
            metadata = {
                "type": "token",
                "n_documents": len(texts),
                "n_tokens": len(vectors),
                "n_unique_tokens": len(set(t for tl in token_lists for t in tl)),
                "embedding_dim": int(self.embedding_dim),
                "model": self.model_name,
                "token_lists": token_lists  # Preserve mapping
            }
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        print(f"✓ Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
        
        return vectors, metadata, texts

    def save_vectors_parquet(self, vectors: np.ndarray, metadata: Dict, 
                            output_path: str):
        """Save vectors to Parquet format (optimized for ML pipelines)."""
        try:
            import pandas as pd
            df = pd.DataFrame(vectors)
            df.to_parquet(output_path, index=False)
            print(f"✓ Vectors saved to {output_path}")
            
            # Save metadata separately
            meta_path = output_path.replace('.parquet', '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to {meta_path}")
        except ImportError:
            print("❌ Please install pandas and pyarrow: pip install pandas pyarrow")

    def save_vectors_numpy(self, vectors: np.ndarray, metadata: Dict, 
                          output_path: str):
        """Save vectors to NumPy format (fast, compact)."""
        np.save(output_path, vectors)
        print(f"✓ Vectors saved to {output_path}")
        
        meta_path = output_path.replace('.npy', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {meta_path}")

    def save_vectors_hdf5(self, vectors: np.ndarray, metadata: Dict, 
                         output_path: str):
        """Save vectors to HDF5 format (efficient for large-scale data)."""
        try:
            import h5py
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('embeddings', data=vectors, compression='gzip')
                f.attrs['embedding_dim'] = self.embedding_dim
                f.attrs['model'] = self.model_name
            print(f"✓ Vectors saved to {output_path}")
            
            meta_path = output_path.replace('.h5', '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to {meta_path}")
        except ImportError:
            print("❌ Please install h5py: pip install h5py")

    def save_vectors(self, vectors: np.ndarray, metadata: Dict, 
                    output_path: str, format: str = "parquet"):
        """
        Save vectors in multiple formats.
        
        Args:
            vectors: Embedding vectors (numpy array).
            metadata: Metadata dictionary.
            output_path: Output file path (without extension if using parquet/numpy/hdf5).
            format: "parquet" (default), "numpy", or "hdf5".
        """
        if format == "parquet":
            if not output_path.endswith('.parquet'):
                output_path += '.parquet'
            self.save_vectors_parquet(vectors, metadata, output_path)
        elif format == "numpy":
            if not output_path.endswith('.npy'):
                output_path += '.npy'
            self.save_vectors_numpy(vectors, metadata, output_path)
        elif format == "hdf5":
            if not output_path.endswith('.h5'):
                output_path += '.h5'
            self.save_vectors_hdf5(vectors, metadata, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def load_vectors(self, file_path: str):
        """Load vectors and metadata from disk."""
        if file_path.endswith('.parquet'):
            import pandas as pd
            vectors = pd.read_parquet(file_path).values
        elif file_path.endswith('.npy'):
            vectors = np.load(file_path)
        elif file_path.endswith('.h5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                vectors = f['embeddings'][:]
        else:
            raise ValueError(f"Unknown file format: {file_path}")
        
        # Load metadata
        meta_path = file_path.replace('.parquet', '_metadata.json') \
                             .replace('.npy', '_metadata.json') \
                             .replace('.h5', '_metadata.json')
        
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"✓ Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
        return vectors, metadata


# Example Usage
if __name__ == "__main__":
    embedder = TokenEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    input_file = "data.txt"
    
    try:
        # Process file
        vectors, metadata, texts = embedder.process_file(input_file, embedding_type="sentence")
        
        # Save in preferred format
        embedder.save_vectors(vectors, metadata, "embeddings", format="parquet")
        # embedder.save_vectors(vectors, metadata, "embeddings", format="numpy")
        # embedder.save_vectors(vectors, metadata, "embeddings", format="hdf5")
        
        print(f"\n✅ Successfully processed {vectors.shape[0]} documents")
    except Exception as e:
        print(f"❌ Error: {e}")
