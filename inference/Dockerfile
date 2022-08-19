FROM python:3.10

WORKDIR /app

ADD requirements.txt requirements.txt
ADD ./inference /app/inference
ADD ./configs /app/configs

RUN python -m pip install -r requirements.txt
EXPOSE 8500


CMD [ "gunicorn", \
    # FastAPI using uvicorn as ASGI framework
    # Swith worker class to Uvicorn to avoid errors
    "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8500", "main:app" \
    ]