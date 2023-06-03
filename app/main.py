import io
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import base64


app = FastAPI()

class Base64Request(BaseModel):
    base64File: str

class Base64Resposne(BaseModel):
    responseCode: int
    responseImage: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/image-detection")
async def image_dectection(request: Base64Request):

    image_data = base64.b64decode(request.base64File)

    response = Base64Resposne(
        responseCode = 200,
        responseImage = base64.b64encode(image_data).decode("utf-8")
        
    ).dict()
    return response


@app.post("/image-detection2")
async def image_detection2(image: UploadFile = File(...)):
    image_data = await image.read()
    file_format = image.content_type.split('/')[-1]
    return StreamingResponse(io.BytesIO(image_data), media_type=f'image/{file_format}')


