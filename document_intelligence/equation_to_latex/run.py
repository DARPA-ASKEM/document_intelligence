from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List
import base64
from texteller.mixed_inference_model import MixedInferenceModel


app = FastAPI()

model = MixedInferenceModel()

class ImageData(BaseModel):
    images: Dict[int, List[str]]  # {page_num: [base64_images]}

@app.post("/predict_latex")
async def predict_latex(data: ImageData):
    latex_results = {}

    for page_num, base64_images in data.images.items():
        latex_results[page_num] = []
        for img_base64 in base64_images: # TODO: use batching
            image_bytes = base64.b64decode(img_base64)
            latex_result = model.predict(image_bytes)
            latex_results[page_num].append(latex_result)

    return JSONResponse(content=latex_results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
