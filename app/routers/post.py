from fastapi import APIRouter, UploadFile, HTTPException
from ..logic.inference import fruit_classifier
from PIL import Image
import numpy as np
import io


router = APIRouter(
    prefix='/classify',
    tags=['Post']
)


@router.post('/')
def classify_img(img:UploadFile):
    # take the image from the payload
    # send it to the inference script
    # return the classified result

    if img.filename.split('.')[-1] in ('jpg', 'jpeg', 'png'):
        pass
    else:
        raise HTTPException(
            status_code=415, detail='Image not found'
        )
    
    image = Image.open(io.BytesIO(img.file.read()))
    image = np.float32(image)

    return fruit_classifier(image)