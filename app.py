from flask import Flask, request, jsonify, render_template
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

from rag_store import ingest_pdf, retrieve_context
from image_ingest import ingest_image, last_extracted_text

import os

app = Flask(__name__)
client = Groq(api_key=GROQ_API_KEY)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "").strip()

    router_prompt = [
        {
            "role": "system",
            "content": (
                "You are a router. Decide which tool to use.\n"
                "Tools:\n"
                "RAG_SEARCH – questions specifically about uploaded PDFs or documents\n"
                "OCR_TEXT – questions about extracted image text\n"
                "CHAT – general knowledge or casual conversation\n\n"
                "Respond with ONLY one word: RAG_SEARCH, OCR_TEXT, or CHAT."
            )
        },
        {"role": "user", "content": query}
    ]

    router_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=router_prompt,
        temperature=0
    )

    tool = router_response.choices[0].message.content.strip()

    if tool == "OCR_TEXT":
        if last_extracted_text:
            return jsonify({"answer": last_extracted_text})
        return jsonify({"answer": "No text extracted from the uploaded image."})


    if tool == "RAG_SEARCH":
        context = retrieve_context(query)


        if not context or len(context.strip()) < 50:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant.\n"
                        "Answer clearly in PLAIN TEXT only.\n"
                        "Formatting rules:\n"
                        "- Do NOT use Markdown, bold, italics, or headings\n"
                        "- Use simple bullet points like: • item\n"
                        "- Keep sentences short and clear\n"
                        "- Avoid long paragraphs unless asked\n"
                    )
                },
                {"role": "user", "content": query}
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )

            return jsonify({"answer": response.choices[0].message.content})

        messages = [
            {
                "role": "system",
                "content": (
                    "You MUST answer ONLY using the document text below.\n"
                    "Formatting rules:\n"
                    "- Use PLAIN TEXT only\n"
                    "- Do NOT use Markdown, bold, italics, or headings\n"
                    "- Use simple bullet points like: • item\n"
                    "- Keep answers short and structured\n"
                    "- If the answer is not found, say:\n"
                    "  The document does not contain this information.\n\n"
                    f"Document Content:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        return jsonify({"answer": response.choices[0].message.content})


    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "Answer clearly in PLAIN TEXT only.\n"
                "Formatting rules:\n"
                "- Do NOT use Markdown, bold, italics, or headings\n"
                "- Use simple bullet points like: • item when helpful and move to next line\n"
                "- Keep sentences short and simple\n"
                "- Avoid unnecessary verbosity\n"
            )
        },
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    return jsonify({"answer": response.choices[0].message.content})


@app.route("/upload/pdf", methods=["POST"])
def upload_pdf():
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    ingest_pdf(path)
    return jsonify({"status": "PDF ingested"})


@app.route("/upload/image", methods=["POST"])
def upload_image():
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    ingest_image(
        path,
        retrieve_context.__globals__["index"],
        retrieve_context.__globals__["documents"]
    )

    return jsonify({"status": "Image ingested (OCR)"})


if __name__ == "__main__":
    app.run(debug=True)
