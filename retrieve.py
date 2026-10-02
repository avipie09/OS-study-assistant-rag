import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="os_documents"
)

# Ask a question
question = "What is deadlock?"

# Convert question into an embedding
question_embedding = model.encode(question).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)

# Display retrieved chunks
print("\nRetrieved chunks:\n")

for i in range(5):
    print("=" * 70)
    print("Result:", i + 1)
    print("Page:", results["metadatas"][0][i]["page"])
    print("Source:", results["metadatas"][0][i]["source"])
    print("Distance:", results["distances"][0][i])
    print("\nText:")
    print(results["documents"][0][i])