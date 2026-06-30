import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text


def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation


def handle_userinput(user_question):

    if st.session_state.conversation is None:
        st.warning("Upload and process PDFs before asking a question.")
        return

    response = st.session_state.conversation(
        {"question": user_question}
    )

    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)

        else:
            with st.chat_message("assistant"):
                st.write(message.content)


def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with Multiple PDFs 📚")

    user_question = st.text_input(
        "Ask a question about your documents:"
    )

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:

        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click Process",
            accept_multiple_files=True
        )

        if st.button("Process"):

            with st.spinner("Processing..."):

                if not pdf_docs:
                    st.warning("Please upload at least one PDF.")
                    return

                # Read PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.warning("No extractable text was found in the uploaded PDFs.")
                    return

                # Split into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create FAISS Vector Store
                vectorstore = get_vectorstore(text_chunks)

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

                st.success("PDFs processed successfully.")


if __name__ == "__main__":
    main()