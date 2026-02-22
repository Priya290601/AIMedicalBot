# AIMedicalBot
An AI-powered Retrieval-Augmented Generation (RAG) System built using Streamlit (Frontend) and Groq LLM + LangChain + FAISS (Backend AI Engine).

This application allows users to:

📄 Load and process PDF documents
🔍 Convert documents into vector embeddings
🤖 Ask context-aware questions
📚 Retrieve relevant document chunks
🧠 Generate accurate answers using Llama 3.3 70B
🚫 Prevent hallucinations (Strict context-based answers)

🏗️ Architecture Overview
🔹 Frontend: Streamlit UI

The frontend is built using Streamlit, which handles:

Chat-based interface

User input (question prompt)

Displaying AI responses

Maintaining chat history (session state)

Streamlit acts as both:

🎨 UI Layer

🔄 Request Handler

🔹 Backend: RAG AI Layer

The backend logic is handled by:

Groq LLM (Llama 3.3 70B)

LangChain Framework

FAISS Vector Database

HuggingFace Embeddings

Backend Responsibilities:

Load stored vector database

Retrieve top-k relevant chunks

Inject retrieved context into prompt

Generate structured answer

Restrict response to document context only

🔄 How Frontend is Connected to Backend

Even though this is a single Python application, it follows a logical frontend-backend separation.

Streamlit handles user interaction, while the RAG pipeline processes and generates answers.

📚 Document Processing Pipeline
🗂️ Step 1: Load PDF Files

PDFs are loaded using DirectoryLoader

Extracted using PyPDFLoader

✂️ Step 2: Create Text Chunks

Uses RecursiveCharacterTextSplitter

Chunk size: 500

Overlap: 50

This ensures better semantic retrieval.

🧠 Step 3: Generate Embeddings

Embedding Model Used:

all-MiniLM-L6-v2

Provider: Hugging Face

Each text chunk is converted into a vector representation.

🗄️ Step 4: Store in Vector Database

Vector Store Used:

FAISS

Embeddings are stored locally inside:

vectorstore/db_faiss
💬 Question Answering Flow
🧾 User Query Flow

1️⃣ User enters question in Streamlit UI
2️⃣ Query is sent to retriever
3️⃣ Top 3 relevant chunks are fetched
4️⃣ Context + Question is passed to LLM
5️⃣ Groq LLM generates structured answer
6️⃣ Response displayed in chat UI

🧠 AI Model Configuration

LLM Provider:

Groq

Model Used:

Llama 3.3 70B Versatile

Configuration:

ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=512
)

Capabilities:

Context-aware reasoning

Long-context handling

Deterministic responses

Reduced hallucination

📦 Tech Stack
Layer	Technology Used
UI	Streamlit
LLM	Groq (Llama 3.3 70B)
Framework	LangChain
Embeddings	HuggingFace
Vector DB	FAISS
Prompt Hub	LangChain Hub
Env Handling	python-dotenv
🚀 How to Run the Project
1️⃣ Clone Repository
2️⃣ Install Dependencies
3️⃣ Set Groq API Key

Option 1 – Environment Variable (Recommended):

Mac/Linux:

export GROQ_API_KEY="your_api_key"

Windows:

set GROQ_API_KEY=your_api_key

Option 2 – .env File:

GROQ_API_KEY=your_api_key
4️⃣ Create Vector Database

Place PDFs inside data/ folder.

Run:

python create_vectorstore.py

This will:

Load PDFs

Split into chunks

Generate embeddings

Store FAISS index

5️⃣ Run Streamlit App
streamlit run medibot.py

Open browser:

http://localhost:8501

Start chatting with your documents 🎉

🔒 Prompt Engineering Strategy

The chatbot is configured to:

Use only retrieved context

Avoid hallucination

Say “I don’t know” if answer not found

Provide direct answers (No small talk)

Custom Prompt:

Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know.
Dont provide anything out of the given context.
🧩 Is This Really Frontend + Backend?

Yes — logically.

Even though it's one Python project:

Streamlit = Frontend Layer

RAG Pipeline = Backend AI Layer

This mimics real-world AI system architecture.
