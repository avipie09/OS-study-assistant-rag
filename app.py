import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from build_db import build_database


load_dotenv()


st.set_page_config(
    page_title="OS Study Assistant",
    page_icon="📚"
)


st.title("📚 Operating Systems Study Assistant")
st.write("Ask questions from the Operating Systems textbook.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


@st.cache_resource
def load_database():
    return build_database()


collection = load_database()


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()


groq_client = Groq(api_key=api_key)


question = st.text_input(
    "Ask an Operating Systems question:"
)


if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        question_embedding = model.encode(
            question
        ).tolist()


        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )


        context = ""

        for i in range(5):

            page = results["metadatas"][0][i]["page"]
            text = results["documents"][0][i]

            context += f"\n[Page {page}]\n{text}\n"


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


        with st.spinner(
            "Searching the textbook and generating answer..."
        ):

            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


        answer = response.choices[0].message.content


        st.subheader("Answer")
        st.write(answer)


        st.subheader("Sources")

        shown_pages = set()

        for i in range(5):

            page = results["metadatas"][0][i]["page"]

            if page not in shown_pages:

                st.write(
                    f"📄 Operating_System.pdf — Page {page}"
                )

                shown_pages.add(page)