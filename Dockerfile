FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libssl-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

VOLUME /app

WORKDIR /app

EXPOSE 5001

CMD ["/bin/bash"]