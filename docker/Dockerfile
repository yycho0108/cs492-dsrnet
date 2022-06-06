FROM nvidia/cuda:11.0.3-devel-ubuntu20.04 as base

# NOTE(ycho): Avoid questions during build process
ENV DEBIAN_FRONTEND=noninteractive
# NOTE(ycho): Avoid images getting bigger than necessary
ENV PIP_NO_CACHE_DIR=1

# Install core packages.
# NOTE(ycho): python3-tk required for matplotlib visualization
RUN apt-get update && \
    apt-get install -y \
    sudo \
    ssh \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install basic python deps.
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade wheel setuptools Cython

# Install python requirements.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt-lists/*

RUN python3 -m pip install open3d

