from components.assets import arrow_circle_icon, github_icon 
from components.chat import chat, chat_form, chat_message
import asyncio
import modal
from fasthtml.common import *
import fastapi
import logging
from transformers import AutoTokenizer
import uuid
from modal import Secret  # Import Secret
from fastlite import Database  # For database operations
from starlette.middleware.sessions import SessionMiddleware  # For session handling
import aiohttp  # For asynchronous HTTP requests
import os
import sqlite3

# Constants
MODELS_DIR = "/llamas_8b"
MODEL_NAME = "Llama-3.1-8B-Instruct" 
FAISS_DATA_DIR = "/faiss_data_pdfs"  # <--- Updated path
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
USERNAME = "c123ian"
APP_NAME = "rag"
DATABASE_DIR = "/db_rag_advan"  # Database directory

db_path = os.path.join(DATABASE_DIR, 'chat_history.db')

# Ensure the directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

# Create the table if it does not exist (here we drop if it exists, for demo)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    DROP TABLE IF EXISTS conversations_history_table_sqlalchemy_v2
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations_history_table_sqlalchemy_v2 (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        top_source_headline TEXT,
        top_source_url TEXT,
        cosine_sim_score REAL, 
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

# Step 2: Initialize FastLite Database connection
db = Database(db_path)
conversations = db['conversations']  # Access the existing table

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Download the model weights
try:
    volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.10") \
    .pip_install(
        "vllm==0.5.3post1",
        "python-fasthtml==0.4.3",
        "aiohttp",          
        "faiss-cpu",        
        "sentence-transformers",
        "pandas",
        "numpy",
        "huggingface_hub",
        "transformers",
        "sqlalchemy"
    )

# Define the FAISS volume (using the new name "faiss_data_pdfs")
try:
    faiss_volume = modal.Volume.lookup("faiss_data_pdfs", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Create the FAISS data volume first by running your PDF script")

# Define the database volume
try:
    db_volume = modal.Volume.lookup("db_data", create_if_missing=True)
except modal.exception.NotFoundError:
    db_volume = modal.Volume.persisted("db_data")

# Define the Modal app
app = modal.App(APP_NAME)

# vLLM server implementation with model path handling
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve_vllm():
    import os
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger
    import fastapi
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import asyncio
    from typing import Optional

    # Function to find the model path by searching for 'config.json'
    def find_model_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    # Function to find the tokenizer path by searching for 'tokenizer_config.json'
    def find_tokenizer_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "tokenizer_config.json" in files:
                return root
        return None

    # Check if model files exist
    model_path = find_model_path(MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MODELS_DIR}")

    # Check if tokenizer files exist
    tokenizer_path = find_tokenizer_path(MODELS_DIR)
    if not tokenizer_path:
        raise Exception(f"Could not find tokenizer files in {MODELS_DIR}")

    print(f"Initializing AsyncLLMEngine with model path: {model_path} and tokenizer path: {tokenizer_path}")

    # Create a FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # Create an `AsyncLLMEngine`, the core of the vLLM server.
    engine_args = AsyncEngineArgs(
        model=model_path,     
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Get model config using the robust event loop handling
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    # Initialize OpenAIServingChat
    request_logger = RequestLogger(max_log_len=256)
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],  
        "assistant",
        lora_modules=None,
        prompt_adapters=None,
        request_logger=request_logger,
        chat_template=None,
    )

    @web_app.post("/v1/completions")
    async def completion_generator(request: fastapi.Request) -> StreamingResponse:
        try:
            # Parse request body
            body = await request.json()
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 100)
            request_id = str(uuid.uuid4())
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
                stop=["User:", "Assistant:", "\n\n"],
            )

            async def generate_text():
                full_response = ""
                last_yielded_position = 0
                assistant_prefix_removed = False
                buffer = ""
                
                async for result in engine.generate(prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        new_text = result.outputs[0].text
                        
                        if not assistant_prefix_removed:
                            new_text = new_text.split("Assistant:")[-1].lstrip()
                            assistant_prefix_removed = True
                        
                        if len(new_text) > last_yielded_position:
                            new_part = new_text[last_yielded_position:]
                            buffer += new_part
                            
                            words = buffer.split()
                            if len(words) > 1:
                                to_yield = ' '.join(words[:-1]) + ' '
                                for punct in ['.', '!', '?']:
                                    to_yield = to_yield.replace(f"{punct}", f"{punct} ")
                                to_yield = ' '.join(to_yield.split())
                                buffer = words[-1]
                                yield to_yield + ' '
                            
                            last_yielded_position = len(new_text)
                        
                        full_response = new_text
                
                if buffer:
                    for punct in ['.', '!', '?']:
                        buffer = buffer.replace(f"{punct}", f"{punct} ")
                    buffer = ' '.join(buffer.split())
                    yield buffer

            return StreamingResponse(generate_text(), media_type="text/plain")
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    return web_app

# FastHTML web interface implementation with RAG
@app.function(
    image=image,
    volumes={FAISS_DATA_DIR: faiss_volume, DATABASE_DIR: db_volume},
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    import faiss
    import os
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi.middleware import Middleware
    from starlette.websockets import WebSocket
    import uuid
    import asyncio
    from sqlalchemy import create_engine, Column, String, DateTime, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import datetime

    SECRET_KEY = os.environ.get('YOUR_KEY')
    if not SECRET_KEY:
        raise Exception("YOUR_KEY environment variable not set.")

    # Updated file paths:
    FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    DATA_PICKLE_PATH = os.path.join(FAISS_DATA_DIR, "data.pkl")

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load new PDF-based DataFrame
    df = pd.read_pickle(DATA_PICKLE_PATH)
    # We assume columns "filename" and "text" exist, as per your PDF script
    docs = df['text'].tolist()

    # Load embedding model
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize FastHTML app
    fasthtml_app, rt = fast_app(
        hdrs=(
            Script(src="https://cdn.tailwindcss.com"),
            Link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
            ),
        ),
        ws_hdr=True,
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=SECRET_KEY,
                session_cookie="secure_session",
                max_age=86400,
                same_site="strict",
                https_only=True
            )
        ]
    )

    # Session-specific messages
    session_messages = {}

    # SQLAlchemy base + model
    Base = declarative_base()

    class Conversation(Base):
        __tablename__ = 'conversations_history_table_sqlalchemy_v2'
        message_id = Column(String, primary_key=True)
        session_id = Column(String, nullable=False)
        role = Column(String, nullable=False)
        content = Column(String, nullable=False)
        top_source_headline = Column(String)
        top_source_url = Column(String)
        cosine_sim_score = Column(Float)  # using Float
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Create a SQLAlchemy engine + session
    db_engine = create_engine(f'sqlite:///{os.path.join(DATABASE_DIR, "chat_history.db")}')
    Session = sessionmaker(bind=db_engine)
    sqlalchemy_session = Session()

    async def load_chat_history(session_id):
        """Load chat history for a session from the database."""
        if not isinstance(session_id, str):
            logging.warning(f"Invalid session_id type: {type(session_id)}. Converting to string.")
            session_id = str(session_id)
        
        if session_id not in session_messages:
            try:
                session_history = sqlalchemy_session.query(Conversation)\
                    .filter(Conversation.session_id == session_id)\
                    .order_by(Conversation.created_at)\
                    .all()
                
                session_messages[session_id] = [
                    {"role": msg.role, "content": msg.content}
                    for msg in session_history
                ]
            except Exception as e:
                logging.error(f"Database error in load_chat_history: {e}")
                session_messages[session_id] = []
        
        return session_messages[session_id]

    @rt("/")
    async def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

        messages = await load_chat_history(session_id)

        return Div(
            H1(
                "Chat with Agony Aunt",
                cls="text-3xl font-bold mb-4 text-white"
            ),
            Div(f"Session ID: {session_id}", cls="text-white mb-4"),
            chat(session_id=session_id, messages=messages),
            Div(
                Span("Model status: "),
                Span("⚫", id="model-status-emoji"),
                cls="model-status text-white mt-4"
            ),
            Div(id="top-sources"),
            cls="flex flex-col items-center min-h-screen bg-black",
        )
    #
    def chat_top_sources(top_sources):
        """Display the filenames of the top sources without URLs."""
        #
        return Div(
        Div(
            Div("Top Sources", cls="text-zinc-400 text-sm font-semibold"),
            Div(
                *[
                    Div(
                        [
                            Span(
                                os.path.basename(source['filename']),
                                cls="text-green-500"
                            ),
                            Span(
                                f" (Page {source['page']})",
                                cls="text-zinc-400"
                            )
                        ],
                        cls="font-mono text-sm"
                    )
                    for source in top_sources
                ],
                cls="flex flex-col items-start gap-2",
            ),
            cls="flex flex-col items-start gap-2",
        ),
        cls="flex flex-col items-start gap-2 p-2 bg-zinc-800 rounded-md",
    )

        

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, session_id: str, send):
        logging.info(f"WebSocket received - msg: {msg}, session_id: {session_id}")
        
        if not session_id:
            logging.error("No session_id received in WebSocket connection!")
            return
        messages = await load_chat_history(session_id)

        response_received = asyncio.Event()

        max_tokens = 6000

        async def update_model_status():
            await asyncio.sleep(3)
            if not response_received.is_set():
                for _ in range(25):
                    if response_received.is_set():
                        break
                    await send(
                        Span(
                            "🟡",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                    if response_received.is_set():
                        break
                    await send(
                        Span(
                            "⚫",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                else:
                    if not response_received.is_set():
                        await send(
                            Span(
                                "🔴",
                                id="model-status-emoji",
                                hx_swap_oob="innerHTML"
                            )
                        )
            if response_received.is_set():
                await send(
                    Span(
                        "🟢",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )
                await asyncio.sleep(600)
                await send(
                    Span(
                        "⚫",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )

        asyncio.create_task(update_model_status())

        messages.append({"role": "user", "content": msg})
        message_index = len(messages) - 1

        # Save user message to DB
        from sqlalchemy.orm import sessionmaker
        new_message = Conversation(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role='user',
            content=msg
        )
        sqlalchemy_session.add(new_message)
        sqlalchemy_session.commit()

        await send(chat_form(disabled=True))
        await send(
            Div(
                chat_message(message_index, messages=messages),
                id="messages",
                hx_swap_oob="beforeend"
            )
        )

        # Retrieve top docs from FAISS
        query_embedding = emb_model.encode([msg], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        K = 2  # Define K 
        distances, indices = index.search(query_embedding, K)
        
        retrieved_paragraphs = []
        top_sources = []

        for i, idx in enumerate(indices[0][:K]):
            similarity_score = float(1 - distances[0][i])  # Convert distance to similarity
            paragraph_text = df.iloc[idx]['text']
            pdf_filename = df.iloc[idx]['filename']
            page_num = df.iloc[idx]['page']
            paragraph_size = df.iloc[idx]['paragraph_size']  # New metadata we're tracking
            
            retrieved_paragraphs.append(paragraph_text)
            top_sources.append({
                'filename': pdf_filename,
                'page': page_num,
                'paragraph_size': paragraph_size,  # Include size in metadata
                'similarity_score': similarity_score
            })
        
        # Construct context
        context = "\n\n".join(retrieved_paragraphs)
    
        def build_conversation(messages, max_length=2000):
            conversation = ''
            total_length = 0
            for message in reversed(messages):
                role = message['role']
                content = message['content']
                message_text = f"{role.capitalize()}: {content}\n"
                total_length += len(message_text)
                if total_length > max_length:
                    break
                conversation = message_text + conversation
            return conversation

        conversation_history = build_conversation(messages)

        def build_prompt(system_prompt, context, conversation_history):
            return f"""{system_prompt}

Context Information:
{context}

Conversation History:
{conversation_history}
Assistant:"""

        system_prompt = (
            "You are an 'Agony Aunt' who helps individuals clarify their options. "
            "Provide thoughtful, empathetic, and helpful responses. "
            "Refer to the provided context for guidance. "
            "Do not mention conversation history directly."
        )

        context = "\n\n".join(retrieved_paragraphs[:2])
        prompt = build_prompt(system_prompt, context, conversation_history)

        print(f"Final Prompt being passed to the LLM:\n{prompt}\n")

        vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": 2000,
            "stream": True
        }

        async with aiohttp.ClientSession() as client_session:
            async with client_session.post(vllm_url, json=payload) as response:
                # Create assistant placeholder
                messages.append({"role": "assistant", "content": ""})
                message_index = len(messages) - 1
                await send(
                    Div(
                        chat_message(message_index, messages=messages),
                        id="messages",
                        hx_swap_oob="beforeend"
                    )
                )

        async with aiohttp.ClientSession() as client_session:
            async with client_session.post(vllm_url, json=payload) as response:
                if response.status == 200:
                    response_received.set()
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            text = chunk.decode('utf-8').strip()
                            if text:
                                if not text.startswith(' ') and messages[message_index]["content"] and not messages[message_index]["content"].endswith(' '):
                                    text = ' ' + text
                                messages[message_index]["content"] += text
                                await send(
                                    Span(
                                        text,
                                        hx_swap_oob="beforeend",
                                        id=f"msg-content-{message_index}"
                                    )
                                )
                    # Save assistant message, referencing top source as the PDF filename
                    new_assistant_message = Conversation(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role='assistant',
                        content=messages[message_index]["content"],
                        top_source_headline=top_sources[0]['filename'],  # show the PDF filename
                        top_source_url=None,  # no URL
                        cosine_sim_score=top_sources[0]['similarity_score']
                    )
                    sqlalchemy_session.add(new_assistant_message)
                    sqlalchemy_session.commit()
                    logging.info(f"Assistant message committed to DB successfully - Content: {messages[message_index]['content'][:50]}...")
                else:
                    error_message = "Error: Unable to get response from LLM."
                    messages.append({"role": "assistant", "content": error_message})
                    await send(
                        Div(
                            chat_message(len(messages) - 1, messages=messages),
                            id="messages",
                            hx_swap_oob="beforeend"
                        )
                    )

        await send(
            Div(
                chat_top_sources(top_sources),
                id="top-sources",
                hx_swap_oob="innerHTML"
            )
        )

        await send(chat_form(disabled=False))

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
