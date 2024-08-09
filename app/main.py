from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import base64
from PIL import Image
import io
import numpy as np
from mtcnn import MTCNN
import uvicorn

app = FastAPI()


class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    gender: str  # 'male' or 'female'


detector_mtcnn = MTCNN()


def decode_image(base64_str):
    header, base64_data = base64_str.split(',')
    image_data = base64.b64decode(base64_data)
    return Image.open(io.BytesIO(image_data))


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def detect_face_mtcnn(image):
    image_np = np.array(image)
    results = detector_mtcnn.detect_faces(image_np)
    if results:
        x, y, width, height = results[0]['box']
        return image.crop((x, y, x + width, y + height))
    return None


def apply_stylegan_aging(face_image, age_factors):
    pass


@app.post("/age-image")
async def age_image(request: ImageRequest):
    try:
        # Decode the base64 image
        image = decode_image(request.image)

        # Detect face using MTCNN
        face_image = detect_face_mtcnn(image)

        if not face_image:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Apply aging transformations
        age_factors = [10, 30, 50, 70]  # Example age factors
        aged_images = apply_stylegan_aging(face_image, age_factors)

        # Encode aged images to base64
        aged_images_base64 = [
            encode_image(img)
            for img in aged_images
        ]

        return {"aged_images": aged_images_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
