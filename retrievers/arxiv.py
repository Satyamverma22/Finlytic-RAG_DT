from langchain_community.retrievers import ArxivRetriever

# create arxiv retriever
retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("large language models")

print("\n===== Arxiv Results =====\n")

for i, doc in enumerate(docs, start=1):
    print(f"Result {i}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)
    print()
