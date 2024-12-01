import faiss
import numpy as np

# Function to search for embeddings in FAISS index
def search_similar_embeddings(query_embedding, index_path="embeddings.index", k=1):
    """
    Search for the most similar embeddings in the FAISS index.
    
    Parameters:
    - query_embedding (np.ndarray): The query embedding (e.g., the text embedding).
    - index_path (str): Path to the saved FAISS index file.
    - k (int): The number of nearest neighbors to retrieve.
    
    Returns:
    - tuple: A tuple containing the distances and indices of the nearest neighbors.
    """
    # Load the FAISS index
    index = faiss.read_index(index_path)

    # Perform the search (get k closest matches)
    D, I = index.search(np.array([query_embedding]), k)

    # Output the results (distances and indices)
    print(f"Distances: {D}")
    print(f"Indices: {I}")

if __name__ == "__main__":
    # Example text embedding (replace with actual embedding)
    query_embedding = np.random.rand(512)  # Example random data, replace with actual generated embedding

    # Perform the search
    search_similar_embeddings(query_embedding, index_path="embeddings.index", k=3)