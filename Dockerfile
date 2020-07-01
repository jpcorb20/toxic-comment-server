FROM python:3.6.11
MAINTAINER jpcorbeil@ledevoir.com

COPY . /toxic-comment-server/

WORKDIR /toxic-comment-server

RUN pip3 install --no-cache-dir -r requirements.txt &&\
    python3 baselines.py &&\
    python3 distilroberta.py

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]