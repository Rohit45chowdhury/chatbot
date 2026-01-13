import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from vector_database import load_faiss_db
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# LLM
# -------------------------------
llm_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------
# FAISS RETRIEVAL
# -------------------------------
def retrieve_docs(query, k=5):
    db = load_faiss_db()
    return db.similarity_search(query, k=k)

def retrieve_all_docs(k=15):
    db = load_faiss_db()
    return db.similarity_search("", k=k)

def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

# -------------------------------
# PROMPTS
# -------------------------------
QA_PROMPT = """
Use the context to answer the question.
If the answer is not present, say "I don't know based on the document."

Question:
{question}

Context:
{context}

Answer:
"""

SUMMARY_PROMPT = """
Summarize the document clearly.

Rules:
- Only use provided context
- Simple language
- Structured format

Context:
{context}

Summary:
"""

# -------------------------------
# SAFE RESPONSE HANDLER 🔥
# -------------------------------
def get_text(response):
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)

# -------------------------------
# Q&A
# -------------------------------
def answer_query(documents, model, query):
    context = get_context(documents)

    if not context.strip():
        return "I don't know based on the document."

    prompt = ChatPromptTemplate.from_template(QA_PROMPT)
    chain = prompt | model

    response = chain.invoke({
        "question": query,
        "context": context
    })

    return get_text(response)

# -------------------------------
# SUMMARY
# -------------------------------
def summarize_pdf(model, documents):
    context = get_context(documents)

    if not context.strip():
        return "Document is empty."

    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    chain = prompt | model

    response = chain.invoke({"context": context})

    return get_text(response)
