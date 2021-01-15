FROM rapidsai/rapidsai:0.16-cuda11.0-runtime-ubuntu18.04-py3.8

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev
