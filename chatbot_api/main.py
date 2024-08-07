import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chatbot

class QueryDTO(BaseModel):
    text: str

app = FastAPI()

print("--- finished initializing ---")


@app.get("/")
def index():
    return {
        "data": "the api is on"
    }


@app.post("/query/")
async def query(query_dto: QueryDTO):
    """returns the entire answer at once (no streaming)"""

    output = chatbot.query(query_dto.text)

    return {
        "question": query_dto.text,
        "answer": output.content
    }


@app.post("/query_stream/")
async def query_stream(query_dto: QueryDTO):
    """streams the answer token by token"""

    output_stream = chatbot.query_stream(query_dto.text)

    return StreamingResponse(output_stream, media_type="text/event-stream")
