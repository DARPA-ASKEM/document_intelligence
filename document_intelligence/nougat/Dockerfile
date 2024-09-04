FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir fastapi uvicorn torch transformers pymupdf pillow

EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
