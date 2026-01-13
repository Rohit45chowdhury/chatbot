import os, re, requests, time
from io import BytesIO
from flask import Flask, render_template, request, flash, url_for, session, redirect
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
import google.genai as genai

# =======================
# RAG IMPORTS
# =======================
from rag_pipeline import (
    answer_query,
    retrieve_docs,
    summarize_pdf,
    llm_model,
    retrieve_all_docs
)

from vector_database import (
    load_pdf,
    create_chunks,
    build_faiss_db,
    reset_faiss_db
)

# =======================
# ENV SETUP
# =======================
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# =======================
# APP SETUP
# =======================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecret")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# =======================
# GEMINI CLIENT
# =======================
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3-flash-preview"

def gemini_qa(prompt):
    try:
        res = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt.strip()
        )
        return res.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {e}"

# =======================
# AI TOOLS
# =======================
def weather_tool(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        w = requests.get(url, timeout=10).json()

        if w.get("cod") != 200:
            return {"type": "error", "message": "City not found"}

        return {
            "type": "weather",
            "city": city.title(),
            "temp": w["main"]["temp"],
            "humidity": w["main"]["humidity"],
            "wind": w["wind"]["speed"],
            "desc": w["weather"][0]["description"]
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}

# =======================
# FIXED CURRENCY
# =======================
def parse_currency(prompt):
    pattern = r"(\d+(?:\.\d+)?)\s*(usd|inr|eur|gbp|jpy|aud|cad)\s*(?:to|in)\s*(usd|inr|eur|gbp|jpy|aud|cad)"
    m = re.search(pattern, prompt.lower())
    if m:
        amt, f, t = m.groups()
        return float(amt), f.upper(), t.upper()
    return None

def currency_tool(amount, f, t):
    try:
        url = f"https://v6.exchangerate-api.com/v6/{CURRENCY_API_KEY}/pair/{f}/{t}/{amount}"
        r = requests.get(url, timeout=10).json()
        if r.get("result") != "success":
            raise Exception("Conversion failed")

        return {
            "type": "currency",
            "amount": amount,
            "from_currency": f,
            "to_currency": t,
            "result": round(r["conversion_result"], 2)
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}

def image_tool(prompt):
    try:
        url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=60)

        img = Image.open(BytesIO(r.content))
        fname = f"gen_{int(time.time())}.png"
        img.save(os.path.join(STATIC_FOLDER, fname))

        return {"type": "image", "url": url_for("static", filename=fname)}
    except Exception as e:
        return {"type": "error", "message": str(e)}

# =======================
# AI AGENT
# =======================
def ai_agent(prompt):
    p = prompt.lower()

    if "weather" in p:
        return weather_tool(p.split("in")[-1].strip())

    cur = parse_currency(prompt)
    if cur:
        return currency_tool(*cur)

    if any(k in p for k in ["image", "draw", "generate image"]):
        return image_tool(prompt)

    return {"type": "text", "message": gemini_qa(prompt)}

# =======================
# ROUTES
# =======================
@app.route("/", methods=["GET", "POST"])
def chat():
    session.setdefault("history", [])

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            flash("Enter message")
            return redirect(url_for("chat"))

        session["history"].append({"role": "user", "message": prompt})
        session["history"].append({"role": "bot", **ai_agent(prompt)})
        session.modified = True
        return redirect(url_for("chat"))

    return render_template("chat.html", history=session["history"])

@app.route("/pdf", methods=["GET", "POST"])
def pdf_page():
    answer = summary = question = None
    session.setdefault("pdf_indexed", False)

    if request.method == "POST":
        action = request.form.get("action")

        # PDF UPLOAD
        if action == "upload":
            pdf = request.files.get("pdf")
            if not pdf or not pdf.filename.endswith(".pdf"):
                flash("Invalid PDF")
                return redirect(url_for("pdf_page"))

            reset_faiss_db()
            path = os.path.join(UPLOAD_FOLDER, secure_filename(pdf.filename))
            pdf.save(path)

            docs = load_pdf(path)
            chunks = create_chunks(docs)
            build_faiss_db(chunks)

            session["pdf_indexed"] = True
            flash("PDF Indexed Successfully")
            return redirect(url_for("pdf_page"))

        # PDF QUESTION (✅ FIXED)
        if action == "question":
            question = request.form.get("question", "").strip()

            if not session.get("pdf_indexed"):
                answer = "❌ Upload PDF first"
            elif not question:
                answer = "❌ Enter a question"
            else:
                try:
                    docs = retrieve_docs(question)
                    if not docs:
                        answer = "❌ Answer not found in document"
                    else:
                        answer = answer_query(docs, llm_model, question)
                except Exception as e:
                    answer = f"❌ Error: {e}"

        # PDF SUMMARY
        if action == "summary":
            if not session.get("pdf_indexed"):
                summary = "Upload PDF first"
            else:
                docs = retrieve_all_docs()
                summary = summarize_pdf(llm_model, docs)

    return render_template("pdf.html", answer=answer, summary=summary, question=question)

# =======================
# RUN
# =======================
if __name__ == "__main__":
    app.run(debug=True)
