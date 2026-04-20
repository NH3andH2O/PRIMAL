FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    gcc \
    git \
    libboost-all-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    make \
    pkg-config \
    sudo \
    tk-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

RUN wget -qO /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && conda clean -afy

COPY .devcontainer/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

RUN /opt/conda/envs/primal/bin/pip install --no-cache-dir tqdm tf-slim

ENV PATH=/opt/conda/envs/primal/bin:/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/primal/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate primal" >> /home/vscode/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/opt/conda/envs/primal/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH' >> /home/vscode/.bashrc \
    && chown -R vscode:vscode /home/vscode /opt/conda

WORKDIR /workspaces/PRIMAL