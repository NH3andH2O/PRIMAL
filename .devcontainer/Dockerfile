FROM mcr.microsoft.com/devcontainers/miniconda:1-3

WORKDIR /workspaces/PRIMAL

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml

RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /home/vscode/.bashrc \
    && echo 'conda activate primal' >> /home/vscode/.bashrc