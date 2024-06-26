from openai import OpenAI
import streamlit as st
from langchain import HuggingFaceHub
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from operator import itemgetter
from utils import message_with_histories, AnswerOnlyParser, format_docs
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['HUGGINGFACEHUB_API_TOKEN']

messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I'm a AI assistant, you can call me Lucas"},
]

prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}
History message: {histories}
Question: {question}
"""

rag_prompt = PromptTemplate(input_variables=["context","question","histories"], template=prompt_template)

llama = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceHub(repo_id=llama)

url_site = "https://lilianweng.github.io/posts/2023-06-23-agent/"


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    url_site = st.text_input("Website url", key="website_url")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Llama")
reset_button_key = "reset_button"
reset_button = st.button("Reset Chat",key=reset_button_key)
if reset_button:
    st.session_state.conversation = None
    st.session_state.chat_history = None
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if url_site:
    loader = WebBaseLoader(url_site)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = (
            {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question"), "histories": itemgetter("histories")}
            | rag_prompt
            | llm
            | AnswerOnlyParser()
            )

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
        
    if not url_site:
        st.info("Please provide url website you would like to ask about.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    question = st.session_state.messages[-1]["content"]
    histories = message_with_histories(messages + st.session_state.messages[:-1])
    response = qa_chain.invoke({"question": question, "histories": histories})
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
