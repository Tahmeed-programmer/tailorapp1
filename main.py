from fastapi import FastAPI
from models9 import Yolov4
import json
import io, base64
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Union

class Item2(BaseModel):
    link:base64

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def hello():
    return {"message":"Hello TutLinks.com"}

@app.post("/h5/")
def image_process(link2:Item2):
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(link2.link, "utf-8"))))
    img.save('my-image.jpeg')
    model5=Yolov4(weight_path='./yolov4.weights',
               class_name_path='./class_names/coco_classes.txt')
    model5.predict('./my-image.jpeg', random_color=True,plot_img=False)
    person=model5.detect2.loc[model5.detect2["class_name"]=="person"]
    height=person["h"]
    width=person["w"]
    return json.dumps({"height":height,"width":width})
