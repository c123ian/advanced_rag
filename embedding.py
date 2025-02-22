import modal
import os
import io
import re
import pickle
import base64
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Create a Modal image that includes required system libraries
pandas_faiss_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "poppler-utils")
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
        "opencv-python-headless",
        "rank-bm25",
        "nltk",
        "pdf2image"
    )
)

app = modal.App("process_pdfs_and_store_embeddings")

FAISS_DATA_DIR = "/faiss_data_pdfs"
PDF_DIR = "/pdfs"
PDF_IMAGES_DIR = "/pdf_images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
faiss_volume = modal.Volume.from_name("faiss_data_pdfs", create_if_missing=True)

def generate_unique_folder_name(pdf_path: str) -> str:
    """Generate a consistent unique folder name based on the PDF path/URL"""
    import hashlib
    # Get just the filename if it's a full path
    filename = os.path.basename(pdf_path)
    # Create a hash of the full path to ensure uniqueness
    hash_obj = hashlib.md5(pdf_path.encode())
    # Combine filename and hash for readability + uniqueness
    return f"{filename.split('.')[0]}_{hash_obj.hexdigest()[:8]}"

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
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    from pdf2image import convert_from_path
    from PIL import Image

    # Download NLTK resources
    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

    # Initialize embedding model once outside the loop
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Create directories for storing processed data
    os.makedirs(os.path.join(FAISS_DATA_DIR, PDF_IMAGES_DIR), exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    all_paragraphs = []
    all_metadata = []
    page_images = {}  # Store paths to page images

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        # Create a unique directory for this PDF's images
        pdf_dir_name = generate_unique_folder_name(pdf_path)
        pdf_images_dir = os.path.join(FAISS_DATA_DIR, PDF_IMAGES_DIR, pdf_dir_name)
        os.makedirs(pdf_images_dir, exist_ok=True)

        # Extract text (including OCR for images)
        loader = PyPDFLoader(
            pdf_path,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=RapidOCRBlobParser(),
        )
        docs = loader.load()

        # Convert PDF pages to images and save them
        def process_page(page_num):
            try:
                # Convert the specific page
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150)
                if not images:
                    return None
                
                image = images[0]
                # Resize to reduce storage needs while keeping quality
                width, height = image.size
                new_width = min(width, 1000)
                new_height = int(height * (new_width / width))
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Save the image
                image_path = os.path.join(pdf_images_dir, f"{page_num}.png")
                resized_image.save(image_path, format='PNG', quality=85)
                
                return image_path
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                return None

        # Process pages in parallel
        print(f"Converting PDF pages to images...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_page = {executor.submit(process_page, i): i for i in range(len(docs))}
            for future in future_to_page:
                page_num = future_to_page[future]
                try:
                    image_path = future.result()
                    if image_path:
                        # Store the image path with PDF name and page number as key
                        image_key = f"{pdf_file}_{page_num}"
                        page_images[image_key] = image_path
                except Exception as e:
                    print(f"Error saving image for page {page_num}: {e}")

        # Process each page for text
        for doc in docs:
            page_num = doc.metadata.get('page', 0)
            
            # Get paragraphs for this page with improved splitting strategy
            page_paragraphs = split_into_paragraphs(
                doc.page_content,
                max_paragraph_size=512,
                min_paragraph_size=100
            )
            
            # Add paragraphs and their metadata
            for paragraph in page_paragraphs:
                all_paragraphs.append(paragraph)
                all_metadata.append({
                    'page': page_num,
                    'source_file': pdf_path,
                    'paragraph_size': len(paragraph),
                    'image_key': f"{pdf_file}_{page_num}"  # Key to retrieve the page image
                })

    # Generate embeddings for all paragraphs at once
    print(f"Generating embeddings for {len(all_paragraphs)} paragraphs...")
    embeddings = embedding_model.encode(
        all_paragraphs,
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create and save FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{FAISS_DATA_DIR}/faiss_index.bin")

    # Create BM25 index
    print("Creating BM25 index...")
    # Tokenize paragraphs for BM25
    tokenized_paragraphs = [word_tokenize(paragraph.lower()) for paragraph in all_paragraphs]
    bm25_index = BM25Okapi(tokenized_paragraphs)
    
    # Save BM25 index
    print("Saving BM25 index...")
    with open(f"{FAISS_DATA_DIR}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    
    # Save tokenized paragraphs (needed for BM25 lookups)
    with open(f"{FAISS_DATA_DIR}/tokenized_paragraphs.pkl", "wb") as f:
        pickle.dump(tokenized_paragraphs, f)

    # Save page image paths
    print("Saving PDF page image paths...")
    with open(f"{FAISS_DATA_DIR}/pdf_page_image_paths.pkl", "wb") as f:
        pickle.dump(page_images, f)

    # Save associated DataFrame with paragraph-level metadata
    print("Saving metadata...")
    df = pd.DataFrame({
        "filename": [os.path.basename(meta['source_file']) for meta in all_metadata],
        "page": [meta['page'] for meta in all_metadata],
        "paragraph_size": [meta['paragraph_size'] for meta in all_metadata],
        "text": all_paragraphs,
        "image_key": [meta['image_key'] for meta in all_metadata],
        "full_path": [meta['source_file'] for meta in all_metadata]
    })
    df.to_pickle(f"{FAISS_DATA_DIR}/data.pkl")

    print("✅ Processing complete! FAISS index, BM25 index, PDF images, and text saved.")
    print(f"Total paragraphs processed: {len(all_paragraphs)}")
    print(f"Average paragraph size: {sum(meta['paragraph_size'] for meta in all_metadata) / len(all_metadata):.2f} characters")
    print(f"Total PDF pages stored as images: {len(page_images)}")

@app.local_entrypoint()
def main():
    process_pdfs_and_store_embeddings.remote()



    
