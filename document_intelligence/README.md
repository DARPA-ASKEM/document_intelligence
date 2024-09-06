# NougatEquation Task

This project implements a FastAPI application to process PDF files, extract images, and recognize LaTeX equations from the images using Hugging Face's VisionEncoderDecoderModel. The application is designed to handle PDF uploads, extract images from the pages, and run them through a model to extract LaTeX mathematical notations.

## Features

- Accepts PDF files via a FastAPI endpoint.
- Extracts images from the PDF.
- Recognizes LaTeX equations in images.
- Processes image batches to ensure efficient resource usage.
- Returns the extracted LaTeX equations in a JSON response.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.8+
- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers)
- `Pillow` (Python Imaging Library for image processing)
- `PyMuPDF` (for PDF processing)
- `fastapi` (web framework for the API)
- `uvicorn` (ASGI server)

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/nougat-equation-task.git
    cd nougat-equation-task
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv nougat_env
    source nougat_env/bin/activate
    ```

3. **Install the dependencies**:

    First, create a `requirements.txt` file:

    ```txt
    torch
    transformers
    Pillow
    PyMuPDF
    fastapi
    uvicorn
    ```

    Then install the dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Start the FastAPI server**:

    You can run the FastAPI server using `uvicorn`:

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

    This will start the server at `http://0.0.0.0:8000`.

2. **API Endpoint**:

    - **POST /process_pdf**: Accepts a PDF file and extracts LaTeX equations from the images in the file.

    Example usage with `curl`:

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/process_pdf' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@example.pdf'
    ```

    The response will be a JSON object with the extracted LaTeX equations.

### Example

Suppose you upload a PDF that contains images of mathematical equations. The API will process the PDF, extract the images, and return LaTeX code for the detected equations in JSON format.

Example response:

```json
{
  "0": ["\\( x^2 + y^2 = z^2 \\)"],
  "1": ["\\[ e = mc^2 \\]"]
}

