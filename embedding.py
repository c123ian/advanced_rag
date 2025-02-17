import modal
import os
import io

# Create a Modal image that includes required system libraries
pandas_faiss_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "faiss-cpu",
        "pandas",
        "numpy",
        "huggingface_hub",
        "sentence-transformers",
        "langchain",
        "langchain-community",
        "pypdf",
        "Pillow",
        "rapidocr-onnxruntime",
        "opencv-python-headless"
    )
)

app = modal.App("process_pdfs_and_store_embeddings")

FAISS_DATA_DIR = "/faiss_data_pdfs"
PDF_DIR = "/pdfs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
faiss_volume = modal.Volume.from_name("faiss_data_pdfs", create_if_missing=True)

@app.function(
    image=pandas_faiss_image,
    mounts=[modal.Mount.from_local_dir("./local_pdfs", remote_path=PDF_DIR)],
    volumes={FAISS_DATA_DIR: faiss_volume},
    timeout=4 * 60 * 60
)
def process_pdfs_and_store_embeddings():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders.parsers import RapidOCRBlobParser
    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    text_data = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing {pdf_file}...")

        # Extract text (including OCR for images)
        loader = PyPDFLoader(
            pdf_path,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=RapidOCRBlobParser(),
        )
        docs = loader.load()
        merged_text = "\n".join(doc.page_content for doc in docs)
        text_data.append(merged_text)

        # Save extracted text as a .txt in the FAISS volume
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"  # Removes ".pdf" and adds ".txt"
        txt_path = f"{FAISS_DATA_DIR}/{txt_filename}"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(merged_text)

    # Generate embeddings
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(text_data, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create a FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, f"{FAISS_DATA_DIR}/faiss_index.bin")

    # Save associated DataFrame
    df = pd.DataFrame({"filename": pdf_files, "text": text_data})
    df.to_pickle(f"{FAISS_DATA_DIR}/data.pkl")

    print("✅ Processing complete! FAISS index and text saved.")

@app.local_entrypoint()
def main():
    process_pdfs_and_store_embeddings.remote()
