# Hugging Face Space (Docker SDK) image for the Streamlit RAG app.
# HF Spaces serve on port 7860 and run as a non-root user (uid 1000).
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install dependencies first for better layer caching.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the app. The uploaded Space repo already excludes venv/secrets/index.
COPY --chown=user . ./

EXPOSE 7860

# On first launch the app builds the Chroma index from data/processed/.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
