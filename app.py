import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
load_dotenv()  # Load only once


# Set environment variables
os.environ['USER_AGENT'] = 'myagent'

# Import necessary libraries
import bs4
from flask import Flask, render_template, jsonify, request
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 


llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,

    timeout=None,
    max_retries=2
    
    
    
    ,
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)


from langchain.document_loaders import WebBaseLoader

url = "https://raw.githubusercontent.com/Dataworld123/Datas.txt/main/README.md"
loader = WebBaseLoader(url)
docs = loader.load()
print(f"Loaded {len(docs)} documents")
print("Document content:", docs[0].page_content[:200])






text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Generated {len(splits)} document splits")
for i, split in enumerate(splits):
    print(f"Split {i}: {split.page_content[:100]}")





for i, doc in enumerate(splits):

    doc.metadata["id"] = f"doc_{i}"
    doc.metadata["source"] = "requirements.txt"
    doc.metadata["chunk_index"] = i


def setup_pinecone_vector_store(documents):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "manoindia-index-new"  # New index name
        
        # Delete existing index if exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name in existing_indexes:
            pc.delete_index(index_name)
            print(f"Deleted existing index: {index_name}")
            import time
            time.sleep(10)
        
        # Create fresh index
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new index: {index_name}")
        
        import time
        time.sleep(10)
        
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Add documents
        print(f"Adding {len(documents)} documents")
        vector_store.add_documents(documents)
        print("Documents added successfully")
        
        return vector_store
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e


vectorstore = setup_pinecone_vector_store(splits)

# Test retrieval immediately
print("Testing retrieval...")
test_docs = vectorstore.similarity_search("flask", k=2)
print(f"Found {len(test_docs)} documents in test")
for doc in test_docs:
    print(f"Test doc: {doc.page_content[:50]}")

retriever = vectorstore.as_retriever()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are DotZoo support assistant for question-answering tasks. "
    "If asked who you are, answer: I am DotZoo support assistant. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User query: {msg}")
    
    # Test direct similarity search
    direct_results = vectorstore.similarity_search(msg, k=3)
    print(f"Direct search found {len(direct_results)} documents")
    
    # Test retriever
    retrieved_docs = retriever.invoke(msg)
    print(f"Retriever found {len(retrieved_docs)} documents")
    
    if not retrieved_docs:
        return jsonify({"response": "No relevant documents found in knowledge base."})
    
    input_data = {"input": msg}
    session_id = "abc123"
    
    response = conversational_rag_chain.invoke(
        input_data,
        config={"configurable": {"session_id": session_id}}
    )
    
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    app.run(debug= True)
