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
    temperature=0
)

# -------------------------------
# FAISS RETRIEVAL (🔥 FIXED)
# -------------------------------
def retrieve_docs(query, k=5):
    faiss_db = load_faiss_db()   # 🔥 LOAD FRESH
    return faiss_db.similarity_search(query, k=k)

def retrieve_all_docs(k=15):
    faiss_db = load_faiss_db()   # 🔥 LOAD FRESH
    return faiss_db.similarity_search("", k=k)

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
- Simple legal language
- Structured format

Context:
{context}

Summary:
"""

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

    return response.content

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
    return response.content
