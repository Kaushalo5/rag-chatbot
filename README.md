\# RAG Chatbot (FastAPI + FAISS)



A Retrieval-Augmented Generation (RAG) chatbot built using \*\*FastAPI\*\*, \*\*FAISS\*\*, and \*\*sentence-transformer embeddings\*\*, with a lightweight frontend UI.



This project demonstrates how to build a \*\*grounded AI system\*\* where answers are generated \*\*only from approved documents\*\*, not from the model’s general knowledge.



---



\## 🚀 Features



\- Document generation and approval workflow

\- Chunking and semantic embeddings

\- FAISS vector database for fast similarity search

\- Context-restricted question answering

\- FastAPI backend

\- Simple dark-mode frontend

\- Modular design (LLM can be swapped easily)



---



\## 🧠 Why RAG?



Large Language Models can hallucinate or answer from general knowledge.



RAG ensures:

\- Answers are grounded in \*\*your data\*\*

\- Better trust and explainability

\- Domain-specific knowledge injection

\- Safer enterprise-style AI systems



---



\## 🏗️ Architecture Overview



Frontend (HTML/CSS/JS)

↓

FastAPI Backend

↓

Chunking \& Embeddings

↓

FAISS Vector Store

↓

Context Retrieval

↓

LLM / Fallback Generator



---



\## ⚙️ How to Run Locally



\### 1. Clone the repository

```bash

git clone https://github.com/Kaushalo5/rag-chatbot.git

cd rag-chatbot



2\. Create virtual environment

python -m venv .venv

.venv\\Scripts\\activate



3\. Install dependencies

pip install -r requirements.txt



4\. Set environment variables



Create a .env file:



GEMINI\_API\_KEY=your\_api\_key\_here



5\. Start backend

cd backend

uvicorn app.main:app --reload





Backend runs at:



http://127.0.0.1:8000



6\. Start frontend

cd frontend

python -m http.server 5500 --bind 127.0.0.1





Frontend runs at:



http://127.0.0.1:5500



🧪 Example Flow



Generate a document draft



Approve and index the document



Ask questions



Answers are generated only from indexed content



If the answer is not found in context, the system replies:



"I don't know."



📌 Future Improvements



Multi-document citation highlighting



Authentication \& user roles



Persistent database for metadata



UI improvements



Cloud deployment



📄 License



This project is for learning and demo purposes.





---



\## ✅ Step 2 — Save \& commit README



After saving, run:



```powershell

git add README.md

git commit -m "Add README with setup and architecture"

git push



