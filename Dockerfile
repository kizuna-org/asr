FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
LABEL maintainer="Hugging Face"

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    git \
    libsndfile1-dev \
    tesseract-ocr \
    espeak-ng \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libnvinfer-dev \
    libnvonnxparsers-dev \
    libcudnn9-dev-cuda-12 \
    libcudnn9-cuda-12 \
    cuda-nvrtc-12-9 \
    cuda-nvrtc-dev-12-9

# Add CUDA libraries to the path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow first with a compatible version
ARG TENSORFLOW='2.16.1'
RUN pip install --no-cache-dir "tensorflow==$TENSORFLOW"

# Install tensorflow-datasets for LJSpeech dataset
RUN pip install --no-cache-dir "tensorflow-datasets>=4.9.0"

# Install tensorflow-probability with compatible version
RUN pip install --no-cache-dir "tensorflow-probability<0.25"

# Install additional dependencies for LJSpeech script
RUN pip install --no-cache-dir "librosa>=0.10.0"
RUN pip install --no-cache-dir "matplotlib>=3.5.0"
RUN pip install --no-cache-dir "numpy>=1.21.0"

# Install pydub
RUN pip install --no-cache-dir "pydub>=0.25.1"

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
# Install transformers without dev-tensorflow to avoid version conflicts
RUN pip install --no-cache-dir -e ./transformers[testing]

RUN pip uninstall -y torch flax
RUN pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python setup.py develop

# Copy local dataset
COPY ./datasets /opt/datasets/

# Copy dataset download script
COPY ./scripts/download_dataset.py /opt/download_dataset.py

# Pre-download the LJSpeech dataset
RUN python /opt/download_dataset.py

# Verify dataset download
RUN ls -la /opt/datasets/ && \
    echo "Dataset directory contents:" && \
    find /opt/datasets -type f -name "*.tfrecord*" | wc -l && \
    echo "Total dataset size:" && \
    du -sh /opt/datasets/

# Copy test script
COPY ./scripts/test_dataset.py /opt/test_dataset.py

# Test dataset loading
RUN python /opt/test_dataset.py

# Create outputs directory for visualizations
RUN mkdir -p /opt/outputs

# Copy LJSpeech learning script
COPY ./scripts/ljspeech_demo.py /opt/ljspeech_demo.py

# Set working directory
WORKDIR /opt

# Default command to run tensorflow learning script
CMD ["python", "ljspeech_demo.py", "--mode", "train"]
