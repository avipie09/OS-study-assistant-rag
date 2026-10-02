import os
import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer


PDF_PATH = "data/Operating_System.pdf"
DB_PATH = "chroma_db"
COLLECTION_NAME = "os_documents"


def build_database():

    # Connect to ChromaDB first
    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # If database already exists, return it immediately
    if collection.count() > 0:

        print(
            "Database already contains",
            collection.count(),
            "documents."
        )

        return collection

    # ------------------------------------------
    # Database is empty — build it
    # ------------------------------------------

    print("Opening PDF...")

    pdf = pymupdf.open(PDF_PATH)

    chunks = []

    chunk_size = 1000
    chunk_overlap = 200

    for page_number in range(len(pdf)):

        page = pdf[page_number]
        text = page.get_text().strip()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number + 1,
                    "source": "Operating_System.pdf"
                })

            start = end - chunk_overlap

    pdf.close()

    print(
        "Total chunks created:",
        len(chunks)
    )

    # ------------------------------------------
    # Load embedding model
    # ------------------------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # ------------------------------------------
    # Prepare data
    # ------------------------------------------

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

    # ------------------------------------------
    # Create embeddings
    # ------------------------------------------

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    # ------------------------------------------
    # Store in ChromaDB
    # ------------------------------------------

    print("Storing data in ChromaDB...")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        "Database created successfully."
    )

    print(
        "Total documents:",
        collection.count()
    )

    return collection


if __name__ == "__main__":
    build_database()