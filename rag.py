import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


# -----------------------------
# 1. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="os_documents"
)


# -----------------------------
# 3. Get question from user
# -----------------------------

question = input("\nAsk an Operating Systems question: ")


# -----------------------------
# 4. Convert question to embedding
# -----------------------------

question_embedding = model.encode(question).tolist()


# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)


# -----------------------------
# 6. Build context
# -----------------------------

context = ""

for i in range(5):
    page = results["metadatas"][0][i]["page"]
    text = results["documents"][0][i]

    context += f"\n[Page {page}]\n{text}\n"


# -----------------------------
# 7. Connect to Groq
# -----------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found.")
    exit()

groq_client = Groq(api_key=api_key)


# -----------------------------
# 8. Create RAG prompt
# -----------------------------

prompt = f"""
You are an Operating Systems study assistant.

Answer the user's question using ONLY the provided context from the
Operating Systems textbook.

If the answer cannot be found in the context, say:
"I could not find this information in the provided textbook."

Do not invent information.

Context:
{context}

Question:
{question}
"""


# -----------------------------
# 9. Ask Groq
# -----------------------------

response = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 10. Display answer
# -----------------------------

answer = response.choices[0].message.content

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)


# -----------------------------
# 11. Display sources
# -----------------------------

print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

for i in range(5):
    page = results["metadatas"][0][i]["page"]
    print(f"- Operating_System.pdf, Page {page}")