
---

#  Multimodal RAG Chatbot

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) chatbot** that supports **text, PDF, and image inputs**. PDF documents and images are processed to extract textual content, which is indexed in a **vector database** for semantic retrieval. User queries are answered by a **large language model (LLM)** using only the retrieved context, ensuring **grounded, accurate, and reliable responses**.



---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-rag-chatbot.git
cd multimodal-rag-chatbot
```

---

### 2️⃣ Create & Activate a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install Tesseract OCR (Required for Image OCR)

**Windows**

* Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* Install Tesseract
* Add `tesseract.exe` to system **PATH**

Verify installation:

```bash
tesseract --version
```

---

### 5️⃣ Configure API Keys

Create a file named **`config.py`** in the project root:

```python
GROQ_API_KEY = "your_groq_api_key"
MODEL_NAME = "llama-3.1-8b-instant"
```

---

### 6️⃣ Run the Application

```bash
python app.py
```

Open in your browser:

```
http://127.0.0.1:5000
```


