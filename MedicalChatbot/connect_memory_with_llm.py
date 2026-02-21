import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()

# STEP 1: SETUP GROQ LLM 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

llm=ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY

)


# STEP 2: CUSTOM PROMPT
#Load databse
DB_FAISS_PATH = "vectorstore\db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#STEP 3: BUILD RAG CHAIN
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#Document combiner chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm,  retrieval_qa_chat_prompt)

#Retrieval chain(retriver+doc combiner)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k':3}), combine_docs_chain)


#Now invoke with a single query
user_query=input("write Query Her: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT:" ,response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")