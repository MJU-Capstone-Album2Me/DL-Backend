import os
from fastapi import FastAPI
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import shutil
import s3utils
import inference


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health-check")
def root():
    return {"Hello": "World"}


@app.post("/image-detection")
async def image_detection2(images: List[UploadFile] = File(...)):

    os.makedirs("../datasets/sample_images", exist_ok=True)

    file_formats = []
    for image in images:
        file_formats.append(image.content_type.split('/')[-1])
        file_path = "../datasets/sample_images/" + image.filename
        with open(file_path, "wb") as f:
            f.write(await image.read())

    #inference.deeplearn()
    images_url = []
    for filename in os.listdir("../datasets/sample_images_convert_resize_results_x2_BSRGANx2.pth"):
        file_path = os.path.join("../datasets/sample_images_convert_resize_results_x2_BSRGANx2.pth", filename)

        if os.path.isfile(file_path):
            upload_url = s3utils.upload_file(file_path, filename)
            images_url.append(upload_url)
    shutil.rmtree("../datasets")

    return {"imagesUrl": images_url}
