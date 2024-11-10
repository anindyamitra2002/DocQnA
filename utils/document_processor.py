from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_and_split_pdfs(pdf_files, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    all_documents = []
    for pdf_file in pdf_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, pdf_file.name)
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            all_documents.extend(chunks)
    return all_documents
