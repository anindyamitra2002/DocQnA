# utils/rag_system.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.document_processor import load_and_split_pdfs
from utils.vector_store import create_vector_store, get_retriever
from langchain_core.runnables import RunnablePassthrough

def initialize_llm(model_name="llama3", temperature=0.8, num_predict=256):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict
    )

def process_documents(pdf_files, k):
    documents = load_and_split_pdfs(pdf_files)
    vector_store = create_vector_store(documents)
    retriever = get_retriever(vector_store, k=k)
    return retriever

def format_retrieved_context(docs):
    # Join all page contents from the retrieved documents
    context = "\n".join(doc.page_content for doc in docs)
    return context

def get_answer(llm, retriever, query):
    # Retrieve relevant documents
    # docs = retriever.invoke(query)
    # print("docs: ", docs)
    # # print("doc type: ", type(vars(docs)))
    # # print("doc attr: ", vars(docs)['content'])
    # # context = vars(docs)['content']
    # context = format_retrieved_context(docs)
    
    # Set up the prompt template with the formatted context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context. If you cannot find the answer in the context, say 'I cannot find the answer in the provided context.'"),
        ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    ])

    # Create the chain with the prompt and llm
    chain = {"context": retriever, "query": RunnablePassthrough()} | prompt | llm 
    
    # Invoke the chain and get the response
    response = chain.invoke(query)
    
    return response.content
