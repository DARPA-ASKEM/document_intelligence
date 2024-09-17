# Table Detection OCR Service

This service provides a FastAPI-based API for detecting tables in PDF documents. It uses a YOLO-based object detection model to locate tables in images and return cropped table images for further processing.

## Features

- **Table Detection**: Automatically detects tables in PDF documents using the YOLO model. Outputs them in a human-readable format with appropriate visual cues.
- **Image Cropping**: Crops out detected tables from the PDF pages and returns them as base64-encoded PNG images.
- **FastAPI-based**: Easy to integrate into existing workflows via API.

  
## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/table-detection-ocr.git
   cd table-detection-ocr
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the service**:
   ```bash
   uvicorn main:app --reload --port 8002
   ```

The service will be available at `http://127.0.0.1:8002`.

## API Endpoints

### **POST /predict**
Upload a PDF file and receive base64-encoded images of detected tables in JSON format.

#### Request
- **Method**: POST
- **Endpoint**: `/predict`
- **Parameters**: 
  - `file`: The PDF file to be uploaded.
  - `padding` (optional): Integer value for padding around detected tables (default is 10 pixels).

#### Example CURL Command

```bash
curl -X 'POST' \
  'http://127.0.0.1:8002/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@yourfile.pdf' \
  -F 'padding=10'
```

#### Response
The response will include the detected tables as base64-encoded PNG images in a JSON structure:

```json
{
  "0": [
    {
      "table_text": "Extracted table text here...",
      "score": 9
    },
    {
      "table_text": "Another table's text...",
      "score": 8
    }
  ]
}
```

### **GET /health**
Check the health status of the service.

#### Request
- **Method**: GET
- **Endpoint**: `/health`

#### Response
```json
{
  "status": "OK"
}
```

## Usage Example

1. Upload a PDF using the `/predict` endpoint.
2. The service detects tables, crops them from the images, and returns them as base64-encoded PNG images in JSON format.
3. You can process the images or decode the base64 strings to display them.

## Environment Variables

- **ASKEM_DOC_AI_API_KEY**: This environment variable is required for accessing the OpenAI API used for further table processing. Be sure to set it in your environment.

```bash
export ASKEM_DOC_AI_API_KEY="your-openai-api-key"
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
