pip install langchain langchain-community transformers faiss-cpu sentence-transformers


import os
# Use os.environ instead of os.environment to access environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


from langchain_community.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-small", huggingfacehub_api_token="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") # Replace with your actual token

# RAG with LangChain and HuggingFace (VS Code Version)

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create pipeline and wrap with LangChain
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Prepare documents
text = """
LangChain is a framework for developing applications powered by LLMs (large language models).
It simplifies the process of integrating language models with external data sources, like databases or documents, and enable more advanced features like memory and document retrieval.
"""
documents = [Document(page_content=text)]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Run query
query = "What is langchain"
result = qa.run(query)
print("Answer:", result)
