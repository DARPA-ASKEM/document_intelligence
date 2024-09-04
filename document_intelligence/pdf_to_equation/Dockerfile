FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn pillow numpy opencv-python-headless onnxruntime paddleocr texteller

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
