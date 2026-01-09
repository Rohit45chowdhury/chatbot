import os, re, requests, time
from io import BytesIO
from flask import Flask, render_template, request, flash, url_for, session, redirect
from dotenv import load_dotenv
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai

# RAG
from rag_pipeline import (
    answer_query, retrieve_docs, summarize_pdf,
    llm_model, retrieve_all_docs
)
from vector_database import (
    upload_pdf, load_pdf, create_chunks,
    build_faiss_db, reset_faiss_db
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

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3-flash-preview"

index_ready = False

# =======================
# GEMINI
# =======================
def gemini_qa(prompt):
    try:
        res = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return res.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {e}"

# =======================
# TOOLS
# =======================
def weather_tool(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    w = requests.get(url).json()

    if w.get("cod") != 200:
        return {"type": "error", "message": "❌ City not found"}

    lat, lon = w["coord"]["lat"], w["coord"]["lon"]
    aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    aqi_data = requests.get(aqi_url).json()
    aqi = aqi_data["list"][0]["main"]["aqi"]

    return {
        "type": "weather",
        "city": city.title(),
        "temp": w["main"]["temp"],
        "humidity": w["main"]["humidity"],
        "wind": w["wind"]["speed"],
        "aqi": aqi,
        "desc": w["weather"][0]["description"]
    }

def parse_currency(prompt):
    match = re.search(
        r"convert\s+(\d+(?:\.\d+)?)\s*([a-zA-Z]{3})\s+to\s+([a-zA-Z]{3})",
        prompt.lower()
    )
    return match.groups() if match else None

def currency_tool(amount, f, t):
    url = f"https://v6.exchangerate-api.com/v6/{CURRENCY_API_KEY}/pair/{f}/{t}/{amount}"
    r = requests.get(url).json()

    if r.get("result") != "success":
        return {"type": "error", "message": "❌ Currency conversion failed"}

    return {
        "type": "currency",
        "amount": amount,
        "from": f,
        "to": t,
        "rate": r["conversion_rate"],
        "result": round(r["conversion_result"], 2)
    }

def image_tool(prompt):
    try:
        url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=60)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            filename = f"generated_{int(time.time())}.png"
            path = f"static/{filename}"
            img.save(path)
            return {"type": "image", "url": url_for("static", filename=filename)}
        return {"type": "error", "message": "❌ Image generation failed"}

    except Exception as e:
        return {"type": "error", "message": str(e)}

def youtube_tool(link):
    try:
        vid = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", link)
        if not vid:
            return {"type": "error", "message": "❌ Invalid YouTube link"}

        transcript = YouTubeTranscriptApi.get_transcript(vid.group(1))
        text = " ".join(i["text"] for i in transcript)
        return {"type": "text", "message": gemini_qa("Summarize this transcript in bullet points:\n" + text)}

    except Exception as e:
        return {"type": "error", "message": str(e)}

# =======================
# AI AGENT
# =======================
def ai_agent(prompt):
    p = prompt.lower()

    if "weather" in p:
        city = p.split("in")[-1].strip()
        return weather_tool(city)

    currency_data = parse_currency(prompt)
    if currency_data:
        amount, f, t = currency_data
        return currency_tool(float(amount), f.upper(), t.upper())

    if any(k in p for k in ["image", "draw", "generate image"]):
        return image_tool(prompt)

    if any(k in p for k in ["youtube", "video"]):
        return youtube_tool(prompt)

    return {"type": "text", "message": gemini_qa(prompt)}

# =======================
# ROUTES
# =======================
@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            session["history"].append({
                "role": "user",
                "type": "text",
                "message": prompt
            })

            result = ai_agent(prompt)
            result["role"] = "bot"
            session["history"].append(result)

            session.modified = True

        return redirect(url_for("chat"))

    return render_template("chat.html", history=session["history"])


@app.route("/pdf", methods=["GET", "POST"])
def pdf_page():
    global index_ready
    answer, summary = None, None

    if request.method == "POST":
        if "pdf" in request.files:
            pdf = request.files["pdf"]
            if pdf.filename:
                reset_faiss_db()
                path = upload_pdf(pdf)
                docs = load_pdf(path)
                chunks = create_chunks(docs)
                build_faiss_db(chunks)
                index_ready = True
                flash("✅ PDF Indexed")

        q = request.form.get("question")
        if q and index_ready:
            docs = retrieve_docs(q)
            answer = answer_query(docs, llm_model, q)

        if request.form.get("summary") and index_ready:
            docs = retrieve_all_docs()
            summary = summarize_pdf(llm_model, docs)

    return render_template("pdf.html", answer=answer, summary=summary)

@app.route("/pdf_chat", methods=["GET", "POST"])
def pdf_chat():
    if "pdf_history" not in session:
        session["pdf_history"] = []

    if request.method == "POST":
        # User asks question
        if "question" in request.form:
            user_q = request.form.get("question")
            if user_q:
                session["pdf_history"].append({"role": "user", "type": "text", "message": user_q})

                if index_ready:
                    docs = retrieve_docs(user_q)
                    answer = answer_query(docs, llm_model, user_q)
                    session["pdf_history"].append({"role": "bot", "type": "text", "message": answer})
                else:
                    session["pdf_history"].append({"role": "bot", "type": "text", "message": "❌ PDF not indexed yet."})

                session.modified = True
                return redirect(url_for("pdf_chat"))

        # Summary request
        if "summary" in request.form:
            if index_ready:
                docs = retrieve_all_docs()
                summary = summarize_pdf(llm_model, docs)
                session["pdf_history"].append({"role": "bot", "type": "text", "message": summary})
            else:
                session["pdf_history"].append({"role": "bot", "type": "text", "message": "❌ PDF not indexed yet."})
            session.modified = True
            return redirect(url_for("pdf_chat"))

    return render_template("pdf_chat.html", history=session["pdf_history"])
    

# =======================
# RUN
# =======================
if __name__ == "__main__":
    app.run(debug=True)
