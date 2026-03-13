FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Generate training data and train the model at build time
RUN mkdir -p trained_models data && \
    python -m app.services.mock_data_generator && \
    python -m app.services.risk_model

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
