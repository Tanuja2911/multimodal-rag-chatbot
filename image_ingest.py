from PIL import Image
import pytesseract
import cv2
import numpy as np

last_extracted_text = ""


def ingest_image(path, index, documents):
    global last_extracted_text

    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        print("Failed to load image:", e)
        last_extracted_text = ""
        return

    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

  
    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    last_extracted_text = text.strip()
    print("OCR TEXT:", last_extracted_text)

    if not last_extracted_text:
        return

    chunks = [
        last_extracted_text[i:i + 500]
        for i in range(0, len(last_extracted_text), 500)
    ]

    from rag_store import get_model
    model = get_model()
    embeddings = model.encode(chunks)

    index.add(np.array(embeddings).astype("float32"))
    documents.extend(chunks)
