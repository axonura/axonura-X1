FROM python:3.8

# Set working directory
WORKDIR /workspace

# Add graphics-drivers PPA and install ubuntu-drivers + NVIDIA/CUDA packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:graphics-drivers/ppa -y \
    && apt-get update \
    && apt-get install -y \
        ubuntu-drivers-common \
        python3-pip python3-full python3-dev \
        nvidia-driver-535 nvidia-utils-535 libnvidia-gl-535  # Latest recommended NVIDIA [web:73]\
        nvidia-cuda-toolkit cuda-toolkit-12-1 cuda-nvcc-12-1 \
        libcudnn8 libcudnn8-dev \
    && ubuntu-drivers autoinstall --gpgpu  # Auto-detect/install best NVIDIA GPU drivers [web:65][web:68]\
    && rm -rf /var/lib/apt/lists/*

# NVIDIA/CUDA env vars for TensorFlow GPU detection
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics
ENV PATH=/usr/local/cuda-12.1/bin:/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-12.1

# Copy files
COPY . /workspace

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Default Bash
CMD ["/bin/bash"]