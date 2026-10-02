import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="OS Study Assistant",
    page_icon="📚"
)


# Title
st.title("📚 Operating Systems Study Assistant")
st.write("Ask questions from the Operating Systems textbook.")


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="os_documents"
)


# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

groq_client = Groq(api_key=api_key)


# Question input
question = st.text_input(
    "Ask an Operating Systems question:"
)


# Ask button
if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        # Convert question into embedding
        question_embedding = model.encode(
            question
        ).tolist()


        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )


        # Build context
        context = ""

        for i in range(5):

            page = results["metadatas"][0][i]["page"]
            text = results["documents"][0][i]

            context += f"\n[Page {page}]\n{text}\n"


        # Create prompt
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


        # Ask Groq
        with st.spinner("Searching the textbook and generating answer..."):

            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


        # Display answer
        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)


        # Display sources
        st.subheader("Sources")

        for i in range(5):

            page = results["metadatas"][0][i]["page"]

            st.write(
                f"📄 Operating_System.pdf — Page {page}"
            )