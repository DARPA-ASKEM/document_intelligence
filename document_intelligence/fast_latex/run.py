from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
from cnstd import LayoutAnalyzer
import base64
import io
from texteller.inference_model import InferenceModel

app = FastAPI()
model = InferenceModel()


class PDFImageExtractor:
    def __init__(self, pdf_bytes):
        self.pdf_bytes = pdf_bytes
        self.images = self._pdf_to_images()

    def _pdf_to_images(self):
        pages = convert_from_bytes(self.pdf_bytes)
        return pages

    def get_images(self):
        return self.images

class ImageAnalyzer:
    def __init__(self):
        self.analyzer = LayoutAnalyzer('mfd')

    def analyze_images(self, images):
        all_detections = []
        for image in images:
            detections = self.analyze_image(image)
            all_detections.append(detections)
        return all_detections

    def analyze_image(self, image):
        return self.analyzer.analyze(image, resized_shape=1024)

    def get_cropped_images(self, images, all_detections, isolated_only=True, padding: int = None):
        cropped_images_dict = {}

        for page_number, detections in enumerate(all_detections):
            cropped_images = []
            image_array = np.array(images[page_number])
            for detection in detections:
                if isolated_only and detection['type'] != 'isolated':
                    continue
                box = detection['box']

                x_coords = box[:, 0]
                y_coords = box[:, 1]
                x_min = int(np.min(x_coords))
                x_max = int(np.max(x_coords))
                y_min = int(np.min(y_coords))
                y_max = int(np.max(y_coords))

                if padding:
                    x_min = max(0, x_min - padding)
                    x_max = min(image_array.shape[1], x_max + padding)
                    y_min = max(0, y_min - padding)
                    y_max = min(image_array.shape[0], y_max + padding)

                cropped_image = image_array[y_min:y_max, x_min:x_max]
                cropped_image_pil = Image.fromarray(cropped_image)

                cropped_images.append(cropped_image_pil)

            cropped_images_dict[page_number] = cropped_images

        return cropped_images_dict


@app.post("/predict")
async def process_and_predict(file: UploadFile = File(...), isolated_only: bool = True, padding: int = 10):
    pdf_bytes = await file.read()
    pdf_extractor = PDFImageExtractor(pdf_bytes)
    images = pdf_extractor.get_images()

    analyzer = ImageAnalyzer()
    detections = analyzer.analyze_images(images)
    cropped_images_dict = analyzer.get_cropped_images(images, detections, isolated_only, padding)

    base64_images_dict = {}
    for page_number, cropped_images in cropped_images_dict.items():
        base64_images_dict[page_number] = []
        for image in cropped_images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images_dict[page_number].append(img_str)

    latex_results = {}
    for page_num, base64_images in base64_images_dict.items():
        latex_results[page_num] = []
        for img_base64 in base64_images:
            image_bytes = base64.b64decode(img_base64)
            latex_result = model.predict(image_bytes)
            latex_results[page_num].append(latex_result)

    return JSONResponse(content=latex_results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
