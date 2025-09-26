import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔑 Hugging Face Token
Access_Token = "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"

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    #repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=Access_Token,
    temperature=0,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\n\nFrom the following text:\n{text}",
    input_variables=["question", "text"]
)
parser = StrOutputParser()

# Streamlit UI
st.title("📄 ANINDYA'S PDF Q&A Assistant")
st.write("Upload PDF(s), then ask your question.")

# File uploader
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

# Collect text from uploaded PDFs
full_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text += "\n".join([doc.page_content for doc in docs])

# Question input
question = st.text_area("Enter your question")

if st.button("Get Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        chain = prompt | model | parser
        result = chain.invoke({"question": question, "text": full_text})
        st.subheader("✅ Answer")
        st.write(result)
