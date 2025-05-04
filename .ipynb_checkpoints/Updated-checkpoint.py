import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
# Retrieve API key from secrets
openai_api_key = st.secrets["openai"]["api_key"]

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Streamlit UI
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")
uploaded_file = st.file_uploader("Upload an article", type="txt")
query_text = st.text_input("Enter your question:", placeholder="Please provide a short summary.", disabled=not uploaded_file)

if uploaded_file and query_text:
    with st.spinner("Calculating..."):
        response = generate_response(uploaded_file, query_text)
        st.info(response)