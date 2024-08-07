from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from config import GOOGLE_API_KEY, PINECONE_API_KEY
import os
import time

print("start")

loader = PyPDFDirectoryLoader("pdf")
data = loader.load()

# print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(text_chunks)
# print(len(text_chunks))
# print(text_chunks[1])

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

result = embeddings.embed_query("start")

# print(result)
print(len(result))

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "nasa"
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

index = pc.Index(index_name)

vectorstore = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)

print(vectorstore)

from langchain.chains import RetrievalQA
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

while True:
    user_input = input(f"Input Prompt: ")
    if user_input.lower() == 'exit':
        print('Exiting')
        break
    if user_input == '':
        continue
    result=qa.invoke(user_input)
    print(f"Answer: {result}")

print("end")