import fitz

pdf_path = "data/Operating_System.pdf"

pdf = fitz.open(pdf_path)

print("Number of pages:", len(pdf))

for page_number in range(len(pdf)):
    page = pdf[page_number]
    text = page.get_text()

    print("\n" + "=" * 50)
    print("PAGE", page_number + 1)
    print("=" * 50)

    print(text[:1000])