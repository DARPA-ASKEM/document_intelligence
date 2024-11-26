import base64
import io
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from openai import OpenAI
from PIL import Image, ImageOps
import numpy as np
from ultralyticsplus import YOLO

app = FastAPI()
model = YOLO('foduucom/table-detection-and-extraction')

TABLE_EXTRACTION_PROMPT = """Please extract the tables from the images and provide the data in a tabular format. Some images may not contain tables, and may contain a mix of figures, graphs and equations. Please ignore these images and give them a score of 0. You will structure your response as a JSON object with the following schema:

'table_text': The text extracted from the table, use visual cues to separate the columns and rows. Ensure that greek characters are preserved, do not swap "α" to "a" for example.
'score': A score from 0 to 10 indicating the quality of the extracted table. 0 indicates that the image does not contain a table, 10 indicates a high-quality extraction.

Begin:
"""

def process_and_send_images(output_dict, prompt):
    openai_api_key = os.getenv("ASKEM_DOC_AI_API_KEY")

    if openai_api_key is None:
        raise ValueError("ASKEM_DOC_AI_API_KEY not found in environment variables. Please set 'ASKEM_DOC_AI_API_KEY'.")

    client = OpenAI(api_key=openai_api_key)

    for page_number, base64_images in output_dict.items():
        for idx, img_b64_str in enumerate(base64_images):
            img_type = "image/png"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},

            )

            print(f"Page {page_number}, Image {idx}: Response from OpenAI:")
            print(response)
            message_content = response.choices[0].message.content
            output_dict[page_number][idx] = message_content
    return output_dict



class PDFImageExtractor:
    def __init__(self, pdf_bytes, resize_to: tuple = None):
        self.pdf_bytes = pdf_bytes
        self.resize_to = resize_to
        self.images = self._pdf_to_images()

    def _pdf_to_images(self):
        pages = convert_from_bytes(self.pdf_bytes)
        if self.resize_to:
            resized_pages = []
            for page in pages:
                page = ImageOps.pad(page, self.resize_to, color='white', method=Image.Resampling.LANCZOS)
                resized_pages.append(page)
            return resized_pages
        else:
            return pages

    def get_images(self):
        return self.images


class TableDetector:
    def __init__(self):
        self.model = model

    def detect_tables(self, image):
        image_np = np.array(image)
        results = self.model.predict(image_np)
        return results

    def detect_tables_in_images(self, images):
        all_detections = []
        for image in images:
            detections = self.detect_tables(image)
            all_detections.append(detections)
        return all_detections

    def get_cropped_images(self, images, all_detections, padding: int = None):
        cropped_images_dict = {}
        for page_number, (image, results) in enumerate(zip(images, all_detections)):
            cropped_images = []
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # box.xyxy[0] contains [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    if padding:
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(image.width, x2 + padding)
                        y2 = min(image.height, y2 + padding)
                    # Crop the image
                    cropped_image = image.crop((x1, y1, x2, y2))
                    cropped_images.append(cropped_image)
            cropped_images_dict[page_number] = cropped_images
        return cropped_images_dict


@app.post("/predict")
async def process_and_predict(file: UploadFile = File(...), padding: int = 10):
    pdf_bytes = await file.read()
    pdf_extractor = PDFImageExtractor(pdf_bytes, resize_to=(1024, 1024))
    images = pdf_extractor.get_images()

    detector = TableDetector()
    detections = detector.detect_tables_in_images(images)
    cropped_images_dict = detector.get_cropped_images(images, detections, padding=padding)

    base64_images_dict = {}
    for page_number, cropped_images in cropped_images_dict.items():
        base64_images_dict[page_number] = []
        for image in cropped_images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images_dict[page_number].append(img_str)

    output_dict = process_and_send_images(base64_images_dict, TABLE_EXTRACTION_PROMPT)
    print(output_dict)
    return JSONResponse(content=output_dict)


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "OK"}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
