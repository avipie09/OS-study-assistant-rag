import pymupdf
import json

pdf_path = "data/Operating_System.pdf"
output_path = "data/chunks.json"

chunk_size = 1000
chunk_overlap = 200

pdf = pymupdf.open(pdf_path)

chunks = []

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
            chunk = {
                "text": chunk_text,
                "page": page_number + 1,
                "source": "Operating_System.pdf"
            }

            chunks.append(chunk)

        start = end - chunk_overlap

pdf.close()

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Chunking completed.")
print("Total chunks:", len(chunks))
print("Saved to:", output_path)

print("\nFirst chunk:")
print("Page:", chunks[0]["page"])
print("Source:", chunks[0]["source"])
print("Text:")
print(chunks[0]["text"])