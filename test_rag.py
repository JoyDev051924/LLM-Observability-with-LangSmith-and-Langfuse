import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load, chunk and index the contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    print("Loading document...")
    docs = loader.load()

    # Split the document into chunks
    print("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings and store them
    print("Creating embeddings...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Set up the retrieval system
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the system with a question
    print("\nTesting RAG system...")
    question = "What is Task Decomposition?"
    print(f"\nQuestion: {question}")
    print("\nAnswer:")
    print(rag_chain.invoke(question))

if __name__ == "__main__":
    main() 