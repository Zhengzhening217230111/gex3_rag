from policy_loader import load_policy_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = load_policy_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ". "]
)
split_docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_it_kb"
)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    print(f"Vector DB built. Total split chunks: {len(split_docs)}")
