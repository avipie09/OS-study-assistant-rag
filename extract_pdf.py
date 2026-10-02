import pymupdf

pdf_path = "data/Operating_System.pdf"

pdf = pymupdf.open(pdf_path)

documents = []

for page_number in range(len(pdf)):
    page = pdf[page_number]

    text = page.get_text().strip()

    if text:
        document = {
            "text": text,
            "page": page_number + 1,
            "source": "Operating_System.pdf"
        }

        documents.append(document)

pdf.close()

print("PDF extraction completed.")
print("Total pages with text:", len(documents))

print("\nFirst page:")
print("Page:", documents[0]["page"])
print("Source:", documents[0]["source"])
print("Text:")
print(documents[0]["text"][:1000])