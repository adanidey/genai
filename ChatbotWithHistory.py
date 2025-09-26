import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 🔑 Your Hugging Face token
Access_Token = "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"

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=Access_Token,
    temperature=0,
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

st.title("💬 Anindya's AI Assistant")

# Display existing chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**AI:** {msg.content}")

# Input box for user
user_input = st.text_input("Enter your prompt:", key="input")

if st.button("Send"):
    if user_input.strip():
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Get AI response
        result = model.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=result.content))

        # Rerun Streamlit to update messages
        st.rerun()
    else:
        st.warning("Please enter a message before sending.")
