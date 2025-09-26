import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

import os
Access_Token = "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"

os.environ["Access_Token"] = "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"

# 🔑 Hugging Face Token
Access_Token = "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"

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=Access_Token,
    temperature=0,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\n\nUsing the context:\n{text}",
    input_variables=["question", "text"]
)
parser = StrOutputParser()

# Streamlit UI
st.title("📄 ANINDYA'S PDF Q&A Assistant")
st.write("Upload PDF(s), then ask your question.")

# File uploader
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

# Collect text from uploaded PDFs
docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

if docs:
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # adjust size as needed
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

# Question input
question = st.text_area("Enter your question")

if st.button("Get Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Retrieve top relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 chunks
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        result = qa_chain.run(question)
        st.subheader("✅ Answer")
        st.write(result)
