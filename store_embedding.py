import faiss
import numpy as np

# Function to store embeddings in FAISS index
def store_embeddings(image_embedding, text_embedding, index_path="embeddings.index"):
    """
    Store the image and text embeddings in a FAISS index.
    
    Parameters:
    - image_embedding (np.ndarray): Image embedding.
    - text_embedding (np.ndarray): Text embedding.
    - index_path (str): Path to save the FAISS index file.
    """
    # Create a FAISS index (L2 distance-based)
    dimension = len(image_embedding)  # Size of embedding vector
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(np.array([image_embedding]))  # Add the image embedding to the index
    index.add(np.array([text_embedding]))   # Add the text prompt embedding to the index

    # Save the FAISS index
    faiss.write_index(index, index_path)
    print(f"Embeddings stored in {index_path}")

if __name__ == "__main__":
    # Example embeddings from previous generation (replace these with actual data)
    image_embedding = np.random.rand(512)  # Example random data, replace with actual generated embedding
    text_embedding = np.random.rand(512)   # Example random data, replace with actual generated embedding

    store_embeddings(image_embedding, text_embedding, index_path="embeddings.index")