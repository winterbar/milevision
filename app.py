from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from chain import get_chat_response

app = FastAPI()

app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/icon", StaticFiles(directory="icon"), name="icon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    query: str
    
@app.post("/messages")
def chat(request:ChatRequest):
    answer = get_chat_response(request.query)
    return {"answer":answer}

@app.get("/")
def get_home():
    return FileResponse("index.html")