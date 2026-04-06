FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_standalone.py .

EXPOSE 7860

COPY app.py .

CMD ["python", "app.py"]
