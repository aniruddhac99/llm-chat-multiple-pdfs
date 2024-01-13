# GUI
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    texts = []

    for pdf in pdf_docs:
        # creates an object with pages
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            texts.append(page.extract_text())

    return "".join(texts)


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # each project or experience is within 700 words
        chunk_overlap=200, # start 100 words before in prev chunk
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={'temperature':0.7, 'max_length':1024})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple pdfs :books:")
    job_description = st.text_input("Ask a question about your documents")

    if job_description:
        handle_user_input(job_description)

    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get resume content
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store = get_vector_store(chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
        

# execute only if application is executed not imported
if __name__ == "__main__":
    main()