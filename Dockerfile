FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    flask \
    ffmpeg-python \
    soundfile \
    numpy==1.26.4 \
    scipy \
    scikit-learn \
    faster-whisper==1.0.3 \
    silero-vad \
    omegaconf \
    hydra-core \
    nemo-toolkit[asr]

COPY . /workspace
