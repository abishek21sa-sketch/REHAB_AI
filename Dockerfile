FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir .
EXPOSE 8010
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "uvicorn", "rehab_ai.web.app:app", "--host", "0.0.0.0", "--port", "8010"]
