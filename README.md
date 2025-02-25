# Advanced RAG Application

This repository contains an advanced Retrieval-Augmented Generation (RAG) application built with Modal, vLLM, and FastHTML/HTMX. The application enables semantic search over PDF documents with visual content preservation, hybrid search capabilities (combining semantic and keyword-based search), and sophisticated reranking.

## Features

- **Hybrid Search**: Combines semantic search (FAISS embeddings) with keyword-based search (BM25) for improved retrieval quality
- **Advanced Reranking**: Uses cross-encoder models to rerank retrieved documents for higher precision
- **PDF Processing**: Extracts and stores both text and visual content from PDFs
- **Interactive UI**: Built with FastHTML/HTMX for a responsive, modern interface
- **Streaming Responses**: Utilizes vLLM for efficient streaming of LLM responses
- **Context Window**: Shows source documents with visual context directly in the interface

## Architecture Overview

![Architecture Diagram](https://github.com/user-attachments/assets/233af5c6-5e30-4f29-89e8-a7c891c4da8a)
)
[source](https://parlance-labs.com/education/rag/ben.html)

The application is structured around several key components:

1. **PDF Processing Pipeline**: Extracts text and images from uploaded PDFs
2. **Embedding Generation**: Creates vector embeddings for document paragraphs
3. **Hybrid Search System**: Combines semantic and keyword search with sophisticated reranking
4. **LLM Integration**: Uses vLLM to serve the Qwen 2.5-7B-Instruct-1M model
5. **Web Interface**: Provides an interactive UI with FastHTML/HTMX

## Technical Implementation

### PDF Processing and Embedding

The system processes PDFs in several stages:

1. PDF upload to a Modal volume
2. Text extraction using PyMuPDFLoader
3. Paragraph splitting with intelligent boundaries
4. Page image extraction and storage
5. Embedding generation using sentence transformers (BAAI/bge-small-en-v1.5)
6. FAISS index creation for vector search
7. BM25 index generation for keyword search

```python
# Core embedding generation
embeddings = embedding_model.encode(all_paragraphs, convert_to_tensor=False, 
                                    show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings, dtype="float32")
faiss.normalize_L2(embeddings)

# FAISS index creation
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
```

### Hybrid Search Implementation

The search system combines multiple approaches for better results:

1. **Semantic Search**: Uses FAISS-indexed embeddings for similarity search
2. **Keyword Search**: Leverages BM25 for term-based matching
3. **Weighted Combination**: Blends both scores for initial candidate selection
4. **Cross-Encoder Reranking**: Refines results using a more sophisticated model

```python
# Hybrid search implementation
query_embedding = emb_model.encode([msg], normalize_embeddings=True).astype('float32')
distances, indices = index.search(query_embedding, K)
tokenized_query = word_tokenize(msg.lower())
bm25_scores = bm25_index.get_scores(tokenized_query)
top_bm25_indices = np.argsort(bm25_scores)[-K:][::-1]

all_candidate_indices = list(set(indices[0].tolist() + top_bm25_indices.tolist()))

# Combine scores with weighting
alpha = 0.6
combined_score = alpha * semantic_scores[idx] + (1 - alpha) * keyword_scores[idx]

# Reranking step
ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', model_type="cross-encoder", verbose=0)
ranked_results = ranker.rank(query=msg, docs=docs_for_reranking)
top_ranked_docs = ranked_results.top_k(3)
```

### LLM Integration with vLLM

The application uses vLLM to serve the Qwen 2.5-7B-Instruct-1M model efficiently:

- **Streaming Responses**: Implements token-by-token streaming for better UX
- **OpenAI-Compatible API**: Uses vLLM's compatibility layer for standard interface
- **Optimized Inference**: Leverages page attention for better performance

```python
engine_args = AsyncEngineArgs(
    model=model_path,
    tokenizer=tokenizer_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=367584
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
```

### Web Interface with FastHTML/HTMX

The interface uses FastHTML with HTMX for a responsive, modern experience:

- **WebSocket Communication**: Real-time updates during query processing
- **Document Visualization**: Displays relevant PDF pages alongside answers
- **Status Indicators**: Shows model processing status
- **Conversation History**: Maintains chat history with session management

## Deployment Instructions

### Prerequisites

- Modal CLI installed and configured
- Python 3.10+
- PDF documents for RAG

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/advanced-rag.git
   cd advanced-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install modal python-fasthtml
   ```

3. **Download the model**:
   ```bash
   python download_model.py
   ```

4. **Upload your PDFs to Modal volume**:
   ```bash
   # Create a local_pdfs directory and add your PDFs
   mkdir -p local_pdfs
   # Copy your PDFs to local_pdfs/
   
   # Upload to Modal volume
   modal volume put faiss_data_pdfs your_document.pdf uploaded_pdfs/
   ```

5. **Process the PDFs and generate embeddings**:
   ```bash
   python embedding.py process
   ```

6. **Run the application**:
   ```bash
   modal deploy app_rag.py
   ```

### Using the Application

1. Access the URL provided by Modal after deployment
2. Enter your query in the chat interface
3. Review the AI response along with source document pages
4. Continue the conversation with follow-up questions

## Technical Notes

### vLLM Considerations

- **Cold Boot Time**: vLLM has a longer initial startup time compared to other solutions
- **Streaming Quirks**: You may notice some word merging during streaming, especially at the beginning of sentences

### Performance Optimization

For better performance, consider:

- Adjusting the `K` parameter for retrieval depth
- Tuning the alpha weight between semantic and keyword search
- Experimenting with different reranker models
- Optimizing chunk sizes for document splitting

## Future Improvements

Potential enhancements include:

- Implementing metadata filtering for more precise retrieval
- Adding multi-modal capabilities for image understanding
- Supporting multi-vector representations per document (ColBERT/ColPali approach)  
- Creating a document management interface
- Adding source validation and citation generation

## References

- [A Hackers' Guide to Language Models](https://www.youtube.com/watch?v=jkrNMKz9pWU)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Rerankers library](https://github.com/AnswerDotAI/rerankers)
- [FAISS Documentation](https://faiss.ai/)
