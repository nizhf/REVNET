FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
LABEL org.opencontainers.image.authors="zhifan.ni@tum.de"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color
ENV TZ="Europe/Berlin"
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN useradd -m revnet && mkdir -p /home/revnet/workspace && chown -R revnet:revnet /home/revnet/workspace
WORKDIR /home/revnet/workspace
COPY --chown=revnet:revnet . .
RUN pip3 install --upgrade pip --no-cache-dir && \
    pip3 install -r requirements.txt --no-cache-dir && \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX" pip3 install --no-build-isolation ./extensions/chamfer3D ./extensions/PyTorchEMD ./extensions/pointnet2_ops_lib --no-cache-dir
USER revnet