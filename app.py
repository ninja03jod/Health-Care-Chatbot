import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load the PDF files from the path using PyPDFLoader
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create ConversationalRetrievalChain instance
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

st.title("HealthCare ChatBot 💬🏥")

def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

def display_chat_history():
    with st.form(key='my_form'):
        user_input = st.text_input("Ask me anything about your Mental Health 👩‍⚕️:", placeholder="How can I improve my mental health?")
        submit_button = st.form_submit_button(label='Send ✉️')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.write("HealthCare ChatBot 🧑🏽‍⚕️: ", output)

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()


  