from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "What is deadlock in an operating system?"
text2 = "Explain the situation where processes wait indefinitely for resources."
text3 = "What is CPU scheduling?"

embedding1 = model.encode([text1])
embedding2 = model.encode([text2])
embedding3 = model.encode([text3])

similarity_12 = cosine_similarity(embedding1, embedding2)[0][0]
similarity_13 = cosine_similarity(embedding1, embedding3)[0][0]

print("Similarity between deadlock questions:", similarity_12)
print("Similarity between deadlock and CPU scheduling:", similarity_13)