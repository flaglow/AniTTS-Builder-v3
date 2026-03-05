FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace
ARG INSTALL_CUML=0

# Change Ubuntu mirror to a reliable one (Cloudflare mirror)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list

# Install dependencies with cache cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y \
    git \
    curl \
    wget \
    vim \
    python3-pip \
    ffmpeg && \
    apt-get clean && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Use local project source from build context
WORKDIR /workspace/AniTTS-Builder-v3
COPY . /workspace/AniTTS-Builder-v3

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch-kmeans hdbscan

# Optional GPU HDBSCAN backend (cuML). Disabled by default because it has strict CUDA/runtime constraints.
RUN if [ "$INSTALL_CUML" = "1" ]; then \
      pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 cupy-cuda12x ; \
    else \
      echo "Skipping cuML installation (INSTALL_CUML=0)."; \
    fi

# Create necessary directories
RUN mkdir -p data/audio_mp3 data/audio_wav data/result data/transcribe data/video \
    module/model/MSST_WebUI module/model/redimmet module/model/whisper

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py"]
