import streamlit as st
import replicate
import os
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from pathlib import Path

from components import summaries_chain, split_documents, create_vectorstore, vectorstore_backed_retriever, create_memory, active_chain_tracing, pdf_loader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from uuid import uuid4
from utils import format_documents

if st.secrets['ACTIVE_TRACING']:
    client = active_chain_tracing(st.secrets['LANGCHAIN_API_KEY'])

# App title
st.set_page_config(page_title="💬 Customize Chatbot")

standalone_question_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""

standalone_question_prompt = PromptTemplate(
    input_variables=['chat_history', 'question'], 
    template=standalone_question_template
)

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context 
    to the `LLM` wihch will answer"""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{summaries}}

{{context}}
</context>

Question: {{question}}

Language: {language}.
"""
    return template

answer_prompt = ChatPromptTemplate.from_template(answer_template())

def on_change_uploading():
    st.session_state.file_uploaded = True
    
st.session_state.chain = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = True

# Set the directory where uploaded files will be saved
TMP_DIR = Path("./data").resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path("./data").resolve().parent.joinpath("data", "vector_stores")
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
    
        
with st.sidebar:
    st.title('💬 Open Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    
    if 'GOOGLE_API_KEY' in st.secrets:
        google_api = st.secrets['GOOGLE_API_KEY']
        
    

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a LLM model', ["Gemini-Flash", 'Llama2-70B'], key='selected_model')
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    if selected_model == 'Gemini-Flash':
        llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', 
                                        temperature=temperature,
                                        top_p=top_p,
                                        safety_settings = {
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
                                        },
                                    )
        
        memory = create_memory()
        
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    
os.environ['REPLICATE_API_TOKEN'] = replicate_api
os.environ['GOOGLE_API_KEY'] = google_api


uploaded_file = st.file_uploader("Choose a PDF file for Asking", type="pdf", on_change= on_change_uploading)

if uploaded_file is not None and st.session_state.file_uploaded:
    
    st.session_state.file_uploaded = False
    file_path = os.path.join(TMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"File uploaded")
    
    with st.spinner("Processing..."):
  
        documents = pdf_loader(file_path)
        s_chain = summaries_chain(llm)
        st.session_state.summary_doc = s_chain.invoke({"documents": format_documents(documents), "language": "english"})
        chunks = split_documents(documents)
        vector_store_google = create_vectorstore(
            documents = chunks,
            google_api_key= google_api,
            vectorstore_name="Vit_All_Google_Embeddings"
        )
        st.session_state.base_retriever_google = vectorstore_backed_retriever(vector_store_google, "similarity", k = 5)
        
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            condense_question_prompt=standalone_question_prompt,
            combine_docs_chain_kwargs={'prompt': answer_prompt},
            condense_question_llm=llm,
            memory=memory,
            retriever = st.session_state.base_retriever_google, 
            llm=llm,
            chain_type= "stuff",
            verbose= False,
            return_source_documents=True   
        )
        print(st.session_state.chain)
            



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

def gemini_generate(prompt_input):
    system_prompt = "You are a helpful assistant."
    dialogue = "Histories messages:\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    response = llm.invoke(f"System prompt: {system_prompt}\n{dialogue}\nQuestion: {prompt_input}")
    return response.content
    

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if selected_model == 'Gemini-Flash' and st.session_state.chain:
                response = st.session_state.chain.invoke({"question": prompt, "summaries": st.session_state.summary_doc})
                response = response['answer']
            elif selected_model == 'Gemini-Flash':
                response = gemini_generate(prompt)
            else: response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)