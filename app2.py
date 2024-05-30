import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import docx
from pptx import Presentation
import glob
from link_parser import LinkParser
import zipfile
import os
import io

link_parser = LinkParser()

def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def getText(filename):
    doc = docx.Document(filename)
    fullText = [para.text for para in doc.paragraphs]
    return '\n'.join(fullText)

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

def get_pptx_text(pptx_files):
    prs = Presentation(pptx_files)
    fullText = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                fullText.append(shape.text)
    return '\n'.join(fullText)

def get_text_from_url(url):
    try:
        return link_parser.extract_text_from_website(url)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please enter a valid URL.")
        return None

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1200, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history in reverse order
    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    api_key = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title="RAG On the Fly ", page_icon=":shark:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = None

    st.header("RAG On the Fly with any document / URL :shark:") 

    message = st.text_input("Ask a question")

    if message:
        handle_userinput(message)

    with st.sidebar:
        st.subheader("Your documents")
        st.subheader('Current version, supports PDF, DOCX, PPTX, ZIPPED or not and URL')
        
        docs = st.file_uploader("Upload a document", accept_multiple_files=True)
        url = st.text_input("Enter a URL")

        if st.button("RAG it now !"):
            with st.spinner("Analyzing, Vectorizing, Retrieving..."):
                raw_text = ""

                if docs:
                    if any(doc.name.endswith(".zip") for doc in docs):
                        for doc in docs:
                            if doc.name.endswith(".zip"):
                                unzip_file(doc, 'temp')
                        all_files = glob.glob('temp/*')
                        docs = [file for file in all_files if os.path.splitext(file)[1] in [".pdf", ".docx", ".pptx"]]
                    else:
                        docs = [doc for doc in docs if os.path.splitext(doc.name)[1] in [".pdf", ".docx", ".pptx"]]

                    for doc in docs:
                        if doc.name.endswith(".pdf"):
                            raw_text += get_pdf_text([doc])
                        elif doc.name.endswith(".docx"):
                            raw_text += getText(doc)
                        elif doc.name.endswith(".pptx"):
                            raw_text += get_pptx_text([doc])

                    st.session_state.conversation = None  # Reset conversation

                elif url:
                    raw_text = get_text_from_url(url)
                    st.session_state.conversation = None  # Reset conversation

                else:
                    st.error("Please upload a PDF, DOCX, PPTX file or a ZIP file containing such file(s) or provide a URL for analysis.")
                    st.stop()

                if not raw_text:
                    st.error("Failed to extract text from the provided documents or URL.")
                    st.stop()

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.write("Preparation completed! \n Ask a question in the chatbox to get started.")

                st.subheader("Manual")
                st.write("1. Upload a PDF, DOCX, or PPTX file or provide a URL for analysis.")
                st.write('Note: you can upload multiple files at once as well as combine different file types and content from URLs.')
                st.write("2. Click the 'Analyze' button to start the analysis.")
                st.write("3. Ask a question in the chatbox to get started.")
                st.write("4. The chatbot will provide answers based on the content of the uploaded document or URL.")

if __name__ == '__main__':
    main()
