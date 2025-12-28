from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

_model = None
documents = []
last_extracted_text = ""

DIMENSION = 384
index = faiss.IndexFlatL2(DIMENSION)


def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def reset_store():
    index.reset()
    documents.clear()


def ingest_pdf(path):
    global last_extracted_text

    reset_store()  

    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    last_extracted_text = text.strip()
    if not last_extracted_text:
        return

    chunks = [
        last_extracted_text[i:i + 500]
        for i in range(0, len(last_extracted_text), 500)
    ]

    model = get_model()
    embeddings = model.encode(chunks)

    index.add(np.array(embeddings).astype("float32"))
    documents.extend(chunks)


def retrieve_context(query, k=3):
    if index.ntotal == 0:
        return ""

    model = get_model()
    query_embedding = model.encode([query])

    _, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    return "\n".join(
        documents[i] for i in indices[0] if i < len(documents)
    )
