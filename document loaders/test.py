from langchain_community.document_loaders import PyPDFLoader

# load pdf
pdf_path = "./document loaders/Financial-Compass-Project.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Total pages: {len(docs)}")

for i, doc in enumerate(docs[:3], start=1):
    print(f"\n===== Page {i} =====")
    print(doc.page_content[:1000])
    print(doc.metadata)
