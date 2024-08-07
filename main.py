import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from config import GOOGLE_API_KEY, PINECONE_API_KEY

def load_and_split_documents(directory):
    loader = PyPDFDirectoryLoader(directory)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(data)

def setup_embeddings():
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def setup_pinecone_index(index_name):
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)

def create_vectorstore(text_chunks, embeddings, index_name):
    return PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)

def setup_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

def main():
    print("start")

    text_chunks = load_and_split_documents("pdf")
    print(text_chunks)

    embeddings = setup_embeddings()
    result = embeddings.embed_query("start")
    print(len(result))

    index_name = "nasa"
    index = setup_pinecone_index(index_name)

    vectorstore = create_vectorstore(text_chunks, embeddings, index_name)
    print(vectorstore)

    qa = setup_qa_chain(vectorstore)

    while True:
        user_input = input("Input Prompt: ")
        if user_input.lower() == 'exit':
            print('Exiting')
            break
        if user_input == '':
            continue
        result = qa.invoke(user_input)
        print(f"Answer: {result}")

    print("end")

if __name__ == "__main__":
    main()