import os
import streamlit as st
from pinecone import Pinecone as PP
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import nest_asyncio

# Set up the event loop
nest_asyncio.apply()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Set environment variables
os.environ['PINECONE_API_KEY'] = 'acd968a8-d607-4881-838a-80ae0cd466a4'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyB1p0yVl-BSf7-AGGTdMBBHb5tjsKlEv4s'

# Set up Streamlit app title
st.title('Smart Document Q&A Assistant')

# Function to load documents from uploaded files
def document_loader(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_loader = PyPDFLoader(uploaded_file.name)
            loaded_docs = file_loader.load()
            documents.extend(loaded_docs)
            
        

            
        
    return documents

# Function to chunk documents
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_splitter.split_documents(docs)
    
    return doc_chunks

def main():
    # Sidebar for file upload
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True,type=['pdf'])

    if uploaded_files:
        # Sidebar options for chunk size and number of results
        chunk_size = st.sidebar.slider('Chunk Size', min_value=200, max_value=1000, value=800, step=100)
        chunk_overlap = st.sidebar.slider('Chunk Overlap', min_value=0, max_value=200, value=50, step=10)
        num_results = st.sidebar.slider('Number of Results to Retrieve', min_value=1, max_value=10, value=2)

        # Load and process documents
        docs = document_loader(uploaded_files)
        documents = chunk_data(docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Initialize the embeddings
        embeddings = PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=os.environ['PINECONE_API_KEY'])

        # Set up Pinecone
        pc = PP(index_name="langchain")
        docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name='langchain')

        # Initialize the language model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.4)
        chain = load_qa_chain(llm, chain_type='stuff')

        # Function to retrieve answers with progress bar
        def retrieve_answers(query):
            progress_bar = st.progress(0)
            
            # Step 1: Retrieve matching documents
            progress_bar.progress(30)
            doc_search = docsearch.similarity_search(query, k=num_results)
            
            # Step 2: Run the QA chain
            progress_bar.progress(60)
            response = chain.run(input_documents=doc_search, question=query)
            
            # Step 3: Finalize progress
            progress_bar.progress(100)
            
            return response

        # User query input
        our_query = st.text_input('Ask Something')

        if st.button('Submit'):
            if our_query:
                st.subheader('**Answer:**')
                answer = retrieve_answers(our_query)
                st.write(answer)
            else:
                st.write("Please enter a query.")

if __name__ == "__main__":
    main()
