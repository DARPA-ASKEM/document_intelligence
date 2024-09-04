FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn pillow numpy opencv-python-headless onnxruntime paddleocr \
    && pip install git+https://github.com/j2whiting/texteller.git

EXPOSE 8001

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
