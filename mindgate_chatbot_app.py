import os
import tempfile
import traceback
import streamlit as st
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import google.api_core.exceptions 
from typing import TypedDict
from vector_store import build_vector_store 

genai.configure(api_key="AIzaSyDHXQN-pqTbRu2RuHBF7hBRsdKnxGdl4eo")
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 300,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
chat_session = model.start_chat(history=[])

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5},
)

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful AI assistant working.

If the user greets you (e.g., 'hi', 'hello', 'namaste', etc.), reply politely with a greeting and follow up with:
  - "How can I assist you today regarding Mindgate Solutions?"

If the user asks about your role or purpose, explain:
  - "I am an AI assistant designed to help users understand Mindgate Solutions' offerings, services, and general information."

You are only allowed to answer questions about Mindgate.

Always respond clearly and professionally.
"""
    ),
    HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}"),
])

def parse_output(response_text: str) -> str:
    cleaned = response_text.strip()
    cleaned = cleaned.replace("**", "__")
    cleaned = cleaned.replace("*", "-")
    return cleaned

def retrieve_context(state):
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    context_docs = "\n".join([doc.page_content for doc in docs])
    memory_text = "\n".join(state.get("memory", []))
    full_context = context_docs + "\n\n" + memory_text if memory_text else context_docs
    return {"context": full_context, "question": query, "memory": state.get("memory", [])}

def call_gemini(state):
    context = state["context"]
    question = state["question"]
    memory = state.get("memory", [])

    if "my name is" in question.lower():
        name_part = question.lower().split("my name is")[-1].strip()
        if name_part:
            name = name_part.split()[0].capitalize()
            if not any(f"User's name is {name}" in mem for mem in memory):
                memory.append(f"User's name is {name}.")

    try:
        prompt = prompt_template.format(context=context, question=question)
        response = chat_session.send_message(str(prompt))
        parsed_response = parse_output(response.text)
        return {"response": parsed_response, "memory": memory}
    
    except google.api_core.exceptions.ResourceExhausted:
        traceback.print_exc()
        st.error("🚫 Gemini API quota exceeded. Please try again later.")
        return {"response": "Gemini API quota exceeded. Please try again later.", "memory": memory}
    
    except Exception as e:
        traceback.print_exc()
        st.error("⚠️ An error occurred while generating a response.")
        return {"response": "Something went wrong. Please try again.", "memory": memory}

class StateSchema(TypedDict):
    question: str
    context: str
    response: str
    memory: list[str]


checkpoint = InMemorySaver()
graph = StateGraph(StateSchema)
graph.add_node("retrieve", RunnableLambda(retrieve_context))
graph.add_node("generate", RunnableLambda(call_gemini))
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
chatbot = graph.compile(checkpointer=checkpoint)


st.set_page_config(page_title="Mindgate AI Chatbot", layout="centered")
st.title("🤖 Mindgate AI Chatbot")


uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files to include your own context:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmp_dir:
        user_paths = []
        for file in uploaded_files:
            file_path = os.path.join(tmp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            user_paths.append(file_path)
        build_vector_store(user_file_paths=user_paths)
        st.success("✅ File(s) uploaded and vector store updated.")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {"memory": []}


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])


user_input = st.chat_input("Ask me anything about Mindgate Solutions...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    st.session_state.chat_state["question"] = user_input
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.chat_state = chatbot.invoke(st.session_state.chat_state, config=config)

    bot_reply = st.session_state.chat_state["response"]
    
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})


















# import streamlit as st
# from langgraph.graph import START, END, StateGraph
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain_core.runnables import RunnableLambda
# from langchain_core.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import google.generativeai as genai
# from typing import TypedDict

# # ✅ Configure Gemini API
# genai.configure(api_key="AIzaSyDHXQN-pqTbRu2RuHBF7hBRsdKnxGdl4eo")
# generation_config = {
#     "temperature": 0.7,
#     "top_p": 0.9,
#     "top_k": 40,
#     "max_output_tokens": 300,
#     "response_mime_type": "text/plain",
# }
# model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
# chat_session = model.start_chat(history=[])

# # ✅ Load embeddings and vector store
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5},
# )

# # ✅ Prompt Template
# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(
#         """You are a helpful AI assistant working.

# If the user greets you (e.g., 'hi', 'hello', 'namaste', etc.), reply politely with a greeting and follow up with:
#   - "How can I assist you today regarding Mindgate Solutions?"

# If the user asks about your role or purpose, explain:
#   - "I am an AI assistant designed to help users understand Mindgate Solutions' offerings, services, and general information."

# You are only allowed to answer questions about Mindgate.

# Always respond clearly and professionally.
# """
#     ),
#     HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}"),
# ])

# # ✅ Output Parser Function
# def parse_output(response_text: str) -> str:
#     """
#     Parses and cleans the Gemini response before displaying it to the user.
#     """
#     cleaned = response_text.strip()
#     cleaned = cleaned.replace("**", "__")
#     cleaned = cleaned.replace("*", "-")

#     return cleaned

# # ✅ LangGraph Functions
# def retrieve_context(state):
#     query = state["question"]
#     docs = retriever.get_relevant_documents(query)
#     context_docs = "\n".join([doc.page_content for doc in docs])
#     memory_text = "\n".join(state.get("memory", []))
#     full_context = context_docs + "\n\n" + memory_text if memory_text else context_docs
#     return {"context": full_context, "question": query, "memory": state.get("memory", [])}

# def call_gemini(state):
#     context = state["context"]
#     question = state["question"]
#     memory = state.get("memory", [])

#     if "my name is" in question.lower():
#         name_part = question.lower().split("my name is")[-1].strip()
#         if name_part:
#             name = name_part.split()[0].capitalize()
#             if not any(f"User's name is {name}" in mem for mem in memory):
#                 memory.append(f"User's name is {name}.")

#     prompt = prompt_template.format(context=context, question=question)
#     response = chat_session.send_message(str(prompt))
#     parsed_response = parse_output(response.text)
#     return {"response": parsed_response, "memory": memory}

# # ✅ Define State Schema
# class StateSchema(TypedDict):
#     question: str
#     context: str
#     response: str
#     memory: list[str]

# # ✅ Build LangGraph
# checkpoint = InMemorySaver()
# graph = StateGraph(StateSchema)
# graph.add_node("retrieve", RunnableLambda(retrieve_context))
# graph.add_node("generate", RunnableLambda(call_gemini))
# graph.add_edge(START, "retrieve")
# graph.add_edge("retrieve", "generate")
# graph.add_edge("generate", END)
# chatbot = graph.compile(checkpointer=checkpoint)

# # ✅ Streamlit App UI
# st.set_page_config(page_title="Mindgate AI Chatbot", layout="centered")
# st.title("🤖 Mindgate AI Chatbot")

# # Initialize session state
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = "user-1"
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "chat_state" not in st.session_state:
#     st.session_state.chat_state = {"memory": []}

# user_input = st.chat_input("Ask me anything about Mindgate Solutions...")

# # Display chat history
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["text"])

# # On user input
# if user_input:
#     st.chat_message("user").markdown(user_input)
#     st.session_state.chat_history.append({"role": "user", "text": user_input})

#     # Prepare and invoke graph
#     st.session_state.chat_state["question"] = user_input
#     config = {"configurable": {"thread_id": st.session_state.thread_id}}
#     st.session_state.chat_state = chatbot.invoke(st.session_state.chat_state, config=config)

#     # Get bot response
#     bot_reply = st.session_state.chat_state["response"]
#     st.chat_message("assistant").markdown(bot_reply)
#     st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})





