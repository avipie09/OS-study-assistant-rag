import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="os_documents"
)

print("ChromaDB working.")
print("Collection:", collection.name)
print("Documents:", collection.count())