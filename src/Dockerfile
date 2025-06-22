# pull official base image
FROM python:3.12-alpine

# set work directory
WORKDIR /usr/src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/requirements.txt


# install dependencies
RUN set -eux \
    && apk add --no-cache --virtual .build-deps build-base \
    python3-dev bash xvfb-run\
    && apk add --no-cache libx11-dev libxrender-dev libxext-dev libpng-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /usr/src/requirements.txt \
    && rm -rf /root/.cache/pip

# copy project
COPY . /usr/src/
