import modal
import os
import io
import re
from typing import List

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

def split_into_paragraphs(text: str, max_paragraph_size: int = 512, min_paragraph_size: int = 100) -> List[str]:
    """
    Split text into paragraphs while preserving natural paragraph boundaries when possible.
    If a paragraph is too long, split on sentence boundaries.
    """
    # First split by clear paragraph markers
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    processed_paragraphs = []
    current_paragraph = []
    current_size = 0
    
    for raw_paragraph in raw_paragraphs:
        # If paragraph itself exceeds max size, split on sentence boundaries
        if len(raw_paragraph) > max_paragraph_size:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', raw_paragraph)
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                if current_size + sentence_len <= max_paragraph_size:
                    current_paragraph.append(sentence)
                    current_size += sentence_len
                else:
                    # Save current paragraph if it meets minimum size
                    if current_size >= min_paragraph_size:
                        processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [sentence]
                    current_size = sentence_len
        else:
            # If adding this paragraph exceeds max size, save current and start new
            if current_size + len(raw_paragraph) > max_paragraph_size:
                if current_size >= min_paragraph_size:
                    processed_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [raw_paragraph]
                current_size = len(raw_paragraph)
            else:
                current_paragraph.append(raw_paragraph)
                current_size += len(raw_paragraph)
    
    # Don't forget the last paragraph
    if current_paragraph and current_size >= min_paragraph_size:
        processed_paragraphs.append(' '.join(current_paragraph))
    
    return processed_paragraphs

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

    # Initialize embedding model once outside the loop
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    all_paragraphs = []
    all_metadata = []

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

        # Process each page
        for doc in docs:
            # Get paragraphs for this page with improved splitting strategy
            page_paragraphs = split_into_paragraphs(
                doc.page_content,
                max_paragraph_size=512,  # Adjust based on your needs
                min_paragraph_size=100
            )
            
            # Add paragraphs and their metadata
            for paragraph in page_paragraphs:
                all_paragraphs.append(paragraph)
                all_metadata.append({
                    'page': doc.metadata.get('page'),
                    'source_file': pdf_path,
                    'paragraph_size': len(paragraph)
                })

    # Generate embeddings for all paragraphs at once
    print(f"Generating embeddings for {len(all_paragraphs)} paragraphs...")
    embeddings = embedding_model.encode(
        all_paragraphs,
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=32  # Adjust based on your available memory
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create and save FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{FAISS_DATA_DIR}/faiss_index.bin")

    # Save associated DataFrame with paragraph-level metadata
    print("Saving metadata...")
    df = pd.DataFrame({
        "filename": [meta['source_file'] for meta in all_metadata],
        "page": [meta['page'] for meta in all_metadata],
        "paragraph_size": [meta['paragraph_size'] for meta in all_metadata],
        "text": all_paragraphs
    })
    df.to_pickle(f"{FAISS_DATA_DIR}/data.pkl")

    print("✅ Processing complete! FAISS index and text saved.")
    print(f"Total paragraphs processed: {len(all_paragraphs)}")
    print(f"Average paragraph size: {sum(meta['paragraph_size'] for meta in all_metadata) / len(all_metadata):.2f} characters")

@app.local_entrypoint()
def main():
    process_pdfs_and_store_embeddings.remote()
