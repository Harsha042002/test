# app.py
import os, json, uuid
from typing import List, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse, StreamingResponse, JSONResponse, Response
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import redis
import dotenv

# RAG imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from contextlib import asynccontextmanager

# ─── ENVIRONMENT & INIT ───────────────────────────────────────────────────────
dotenv.load_dotenv()
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
BASE_URL_CUSTOMER = os.getenv("BASE_URL_CUSTOMER", "https://api-stage.freshbus.com")
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_PW          = os.getenv("REDIS_PW", None)
REDIS_SSL         = os.getenv("REDIS_SSL", "false").lower() == "true"

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is required")

genai.configure(api_key=GOOGLE_API_KEY)

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "freshbus_info")
QDRANT_PATH       = os.getenv("QDRANT_PATH", "./qdrant_data")



# ─── BUILD RAG ON STARTUP ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, rag_chain

    with open("data/info.txt", "r", encoding="utf-8") as f:
        info = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    chunks = splitter.split_text(info)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
    os.makedirs(QDRANT_PATH, exist_ok=True)
    vector_db = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        path=QDRANT_PATH
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1
    )
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k":10})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )
    print("✅ RAG chain initialized")
    yield

app = FastAPI(lifespan=lifespan)

# # ─── STATIC + CORS ───────────────────────────────────────────────────────────
# app.mount("/static", StaticFiles(directory="static", html=True), name="static")
# @app.get("/", include_in_schema=False)
# async def root():
#     return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── REDIS CLIENT ─────────────────────────────────────────────────────────────
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PW, ssl=REDIS_SSL, decode_responses=True
)

# ─── Pydantic MODELS ──────────────────────────────────────────────────────────
class OTPRequest(BaseModel):
    mobile: str

class VerifyOTPRequest(OTPRequest):
    otp: int
    deviceId: str

class QueryRequest(BaseModel):
    query:      str
    id:         int
    name: Optional[str] = None  
    mobile:     str
    session_id: Optional[str] = None

# ─── AUTH PROXY ENDPOINTS ────────────────────────────────────────────────────
@app.post("/auth/sendotp", status_code=201)
async def send_otp(payload: OTPRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL_CUSTOMER}/auth/sendotp", json=payload.dict())
    return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers)

@app.post("/auth/resendotp", status_code=201)
async def resend_otp(payload: OTPRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL_CUSTOMER}/auth/resendotp", json=payload.dict())
    return Response(content=resp.content, status_code=resp.status_code, headers=resp.headers)

@app.post("/auth/verifyotp", status_code=201)
async def verify_otp(payload: VerifyOTPRequest):
    # 1) proxy to upstream
    async with httpx.AsyncClient() as client:
        upstream = await client.post(
            f"{BASE_URL_CUSTOMER}/auth/verifyotp",
            json=payload.dict()
        )
    # 2) collect the two Set-Cookie headers
    set_cookies = upstream.headers.get_list("set-cookie")
    # 3) try to parse whatever JSON body upstream sent
    try:
        data = upstream.json()
    except ValueError:
        data = {}
    # 4) optionally pull tokens out of the cookie strings
    for cookie_str in set_cookies:
        if cookie_str.startswith("access_token="):
            token = cookie_str.split(";",1)[0].split("=",1)[1]
            data["token"] = token
        if cookie_str.startswith("refresh_token="):
            rt = cookie_str.split(";",1)[0].split("=",1)[1]
            data["refresh_token"] = rt
    # 5) build our JSONResponse (so client sees JSON + cookies)
    out = JSONResponse(content=data, status_code=upstream.status_code)
    for c in set_cookies:
        out.headers.append("set-cookie", c)
    return out

@app.get("/auth/logout")
async def logout(request: Request):
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(f"{BASE_URL_CUSTOMER}/auth/logout", cookies=request.cookies)
    except:
        pass
    out = JSONResponse({"message":"Logged out"}, status_code=200)
    for name in ("access_token","refresh_token"):
        out.delete_cookie(key=name, path="/", secure=True, httponly=True, samesite="none")
    return out

@app.get("/auth/refresh-token")
async def refresh_token(request: Request):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL_CUSTOMER}/auth/refresh-token", cookies=request.cookies)
    set_cookies = resp.headers.get_list("set-cookie")
    out = Response(content=resp.content, status_code=resp.status_code)
    for c in set_cookies:
        out.headers.append("set-cookie", c)
    return out

@app.get("/auth/profile")
async def profile(request: Request):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL_CUSTOMER}/profile", cookies=request.cookies)
    out = JSONResponse(content=resp.json(), status_code=resp.status_code)
    for sc in resp.headers.get_list("set-cookie"):
        out.headers.append("set-cookie", sc)
    return out

# ─── CONVERSATION LIST & DELETE ───────────────────────────────────────────────
@app.get("/conversations")
async def list_conversations(user_id: int):
    """
    GET /conversations?user_id=123
    returns { conversations: [ { session_id, preview }, … ] }
    """
    pattern = f"convos:{user_id}:*"
    keys = redis_client.keys(pattern)
    convos = []
    for key in keys:
        _, _, sid = key.partition(f"convos:{user_id}:")
        last_entries = redis_client.lrange(key, -2, -1)
        preview = json.loads(last_entries[-1])["content"] if last_entries else ""
        convos.append({"session_id": sid, "preview": preview})
    return JSONResponse(content={"conversations": convos})

@app.delete("/conversations")
async def delete_conversation(user_id: int, session_id: str):
    """
    DELETE /conversations?user_id=123&session_id=abc
    """
    key = f"convos:{user_id}:{session_id}"
    redis_client.delete(key)
    return JSONResponse(content={"deleted": True})

# ─── /query (refresh + RAG + Redis + 2×Set-Cookie) ─────────────────────────────
@app.post("/query")
async def query(req: QueryRequest, request: Request):
    # 1) refresh tokens
    async with httpx.AsyncClient() as client:
        rt = await client.get(f"{BASE_URL_CUSTOMER}/auth/refresh-token", cookies=request.cookies)
    new_cookies = rt.headers.get_list("set-cookie")

    # 2) normalize user data
    user_id = req.id
    name = None
    if req.name:
        stripped = req.name.strip()
        name = stripped if stripped else None
    mobile = req.mobile.strip()

    # 3) key per‐user per‐session
    session_id = req.session_id or str(uuid.uuid4())
    key        = f"convos:{user_id}:{session_id}"

    # on first turn store meta
    if req.session_id is None:
        redis_client.rpush(key, json.dumps({"role":"meta","user":{"id":user_id,"name":name,"mobile":mobile}}))

    # append user & assistant
    redis_client.rpush(key, json.dumps({"role":"user","content":req.query}))
    answer = rag_chain.invoke(req.query)["result"]
    redis_client.rpush(key, json.dumps({"role":"assistant","content":answer}))

    # 4) stream + attach new cookies
    def streamer():
        for w in answer.split():
            yield w + " "
    out = StreamingResponse(streamer(), media_type="text/plain", headers={"X-Session-ID":session_id})
    for c in new_cookies:
        out.headers.append("set-cookie", c)
    return out

@app.get("/history")
async def history(user_id: int, session_id: str):
    key     = f"convos:{user_id}:{session_id}"
    entries = redis_client.lrange(key, 0, -1)
    history = [json.loads(e) for e in entries]
    return JSONResponse(content={"history": history})

@app.get("/health")
async def health():
    return {"status":"healthy", "vector_db_initialized": vector_db is not None}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)