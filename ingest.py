import json
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load the chunks
# -----------------------------

with open("data/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Total chunks loaded:", len(chunks))


# -----------------------------
# 2. Load the embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# -----------------------------
# 3. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="os_documents"
)

print("ChromaDB connected.")


# -----------------------------
# 4. Prepare the data
# -----------------------------

texts = []
ids = []
metadatas = []

for i, chunk in enumerate(chunks):

    texts.append(chunk["text"])

    ids.append(f"chunk_{i}")

    metadatas.append({
        "page": chunk["page"],
        "source": chunk["source"]
    })


# -----------------------------
# 5. Create embeddings
# -----------------------------

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings created.")


# -----------------------------
# 6. Store everything in ChromaDB
# -----------------------------

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Data successfully stored in ChromaDB.")
print("Total documents in database:", collection.count())