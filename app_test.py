
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient
import base64
import hashlib
  

google_api_key = st.secrets['google_api_key']
tvly_api_key = st.secrets['tvly_api_key']
openai_api_key = st.secrets['openai_api_key']

web_tool_search = TavilyClient(api_key=tvly_api_key)

st.set_page_config(
    page_title="AI Professor",
    page_icon="👨‍🏫",
    layout="wide"  
)


st.markdown("""
    <style>
    /* Main container responsiveness */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Sidebar responsiveness */
    .sidebar .sidebar-content {
        width: 100%;
    }
    
    /* Chat message containers */
    .stChatMessage {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* PDF viewer responsiveness */
    iframe {
        width: 100% !important;
        height: 70vh !important;
        min-height: 400px;
    }
    
    /* Responsive text sizing */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        p {
            font-size: 0.9rem !important;
        }
    }
    
    /* Custom sidebar width control */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 300px;
        max-width: 500px;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px;
    }
    
    /* Make buttons more touch-friendly on mobile */
    .stButton button {
        width: 100%;
        margin: 0.5rem 0;
        min-height: 44px;  /* Better touch targets */
    }
    
    /* Improve file uploader responsiveness */
    .stFileUploader {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


st.title("👨‍🏫 AI Professor")

def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, list):
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    else:
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the document provided:

    Context: {context}
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
    base_url = "https://api.groq.com/openai/v1",
    openai_api_key = openai_api_key,
    model_name = "llama-3.1-8b-instant",
    temperature=1,
    max_tokens=1024
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_query)

    context = "\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "user_question": user_query,
    })

def get_youtube_url(query):
    response = web_tool_search.search(
        query=query,
        search_depth="basic",
        include_domains=["youtube.com"],
        max_results=1
    )
    
    for result in response['results']:
        if 'youtube.com/watch' in result['url']:
            return result['url']
    
    return None


def displayPDF(uploaded_file):
    if isinstance(uploaded_file, list):
        for uploaded_file in uploaded_file:
            bytes_data = uploaded_file.getvalue()
            base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
            pdf_display = f'''
                <div style="width: 100%; height: 100%; display: flex; justify-content: center;">
                    <iframe 
                        src="data:application/pdf;base64,{base64_pdf}" 
                        style="width: 100%; height: 70vh; min-height: 400px; border: none;"
                        type="application/pdf">
                    </iframe>
                </div>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)

def get_pdfs_hash(pdf_docs):
    combined_hash = hashlib.md5()
    if isinstance(pdf_docs, list):
        for pdf in pdf_docs:
            content = pdf.read()
            combined_hash.update(content)
            pdf.seek(0)  
    else:
        content = pdf_docs.read()
        combined_hash.update(content)
        pdf_docs.seek(0)  
    return combined_hash.hexdigest()

with st.sidebar:
    st.title("Menu:")
    

    col1, col2 = st.columns(2)
    

    pdf_docs = st.file_uploader(
        "Upload your PDF Files",
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    

    with st.container():
        quizz_button = st.button("🗒️ Make a quizz", type="primary", use_container_width=True)
        video_button = st.button("📺 Search a video", type="secondary", use_container_width=True)
        view = st.toggle("👁️ View PDF", value=False)
        if view and pdf_docs:
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  
            displayPDF(pdf_docs)
            st.markdown("""
                <style>
                    /* Default styles for larger screens */
                    section[data-testid="stSidebar"] {
                        width: 500px !important;
                        background-color: #f0f2f6;
                    }
                    
                    section[data-testid="stSidebar"] > div {
                        width: 100% !important;
                        max-width: 800px;
                        padding: 1rem;
                        background-color: #f0f2f6;
                    }
                    
                    section[data-testid="stSidebar"] .block-container {
                        width: 100% !important;
                        max-width: 500px;
                        background-color: #f0f2f6;
                    }
                    
                    /* Responsive styles for different screen sizes */
                    @media (max-width: 1200px) {
                        section[data-testid="stSidebar"] {
                            width: 400px !important;
                        }
                        
                        section[data-testid="stSidebar"] > div {
                            max-width: 600px;
                        }
                        
                        section[data-testid="stSidebar"] .block-container {
                            max-width: 400px;
                        }
                    }
                    
                    @media (max-width: 992px) {
                        section[data-testid="stSidebar"] {
                            width: 300px !important;
                        }
                        
                        section[data-testid="stSidebar"] > div {
                            max-width: 400px;
                        }
                        
                        section[data-testid="stSidebar"] .block-container {
                            max-width: 300px;
                        }
                    }
                    
                    @media (max-width: 768px) {
                        section[data-testid="stSidebar"] {
                            width: 100% !important;
                            min-width: 100%;
                        }
                        
                        section[data-testid="stSidebar"] > div {
                            width: 100% !important;
                            max-width: 100%;
                            padding: 0.5rem;
                        }
                        
                        section[data-testid="stSidebar"] .block-container {
                            width: 100% !important;
                            max-width: 100%;
                        }
                        
                        /* Adjust PDF viewer for mobile */
                        iframe {
                            height: 50vh !important;
                            min-height: 300px;
                        }
                    }
                    
                    /* Improve PDF viewer container */
                    .pdf-container {
                        width: 100%;
                        height: 100%;
                        margin: 0 auto;
                        overflow: hidden;
                    }
                    
                    /* Ensure content doesn't overflow */
                    .main .block-container {
                        max-width: 100%;
                        padding: 1rem;
                        box-sizing: border-box;
                    }
                </style>
            """, unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Chatbot professor assistant. How can I help you?"),
    ]


chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)


user_query = st.chat_input("Type your message here...")


if pdf_docs:
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.success("The document is loaded")


if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query, unsafe_allow_html=True)
    
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = get_response(user_query, st.session_state.chat_history)
            st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))

if quizz_button:
    with st.spinner("Generating quiz..."):
        quiz_prompt = """
        Based on the document content, create a quiz with 5 multiple choice questions.
        For each question:
        1. Ask a clear, specific question
        2. Provide 4 options labeled A, B, C, D
        3. Make sure the options are plausible but distinct
        4. Don't reveal the correct answer

        Format each question like this:
        Question X:
        **A)**
        **B)**
        **C)**
        **D)**
        """
        with st.chat_message("AI"):
            response = get_response(quiz_prompt, st.session_state.chat_history)
            st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))


if video_button:
    with st.spinner("Searching for relevant video..."):
        video_prompt = """
        Extract the main topic and key concepts from the document or from the last conversation in 3-4 words maximum.
        Focus on the core subject matter only.
        Do not include any additional text or explanation.
        Example format: "machine learning neural networks" or "quantum computing basics"
        """
        with st.chat_message("AI"):
            response = get_response(video_prompt, st.session_state.chat_history)
            youtube_url = get_youtube_url(f"Course on {response}")
            if youtube_url:
                st.write(f"📺 Here's a video about {response}:")
                st.video(youtube_url)
                video_message = f"📺 Here's a video about {response}:\n{youtube_url}"
                st.session_state.chat_history.append(AIMessage(content=video_message))


if "current_pdfs_hash" not in st.session_state:
    st.session_state.current_pdfs_hash = None

if pdf_docs:
    new_hash = get_pdfs_hash(pdf_docs)
    if new_hash != st.session_state.current_pdfs_hash:
        with st.spinner("Updating documents..."):
            text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.session_state.current_pdfs_hash = new_hash
            st.success("Documents have been updated!")
