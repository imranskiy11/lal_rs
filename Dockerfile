FROM nvcr.io/nvidia/pytorch:20.10-py3

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
WORKDIR /lal_tsms
RUN apt update && apt install -yqq git
COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["pwd"]