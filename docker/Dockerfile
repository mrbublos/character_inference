FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1


ENV TORCH_CUDA_ARCH_LIST="Turing"
ENV PATH="/opt/conda/bin:/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.2/compat/:/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="video,compute,utility"
ENV CUDA_HOME='/usr/local/cuda'
ENV BASE_DIR=/runpod-volume


RUN git clone https://github.com/mrbublos/character_inference.git
WORKDIR /app/character_inference
COPY requirements_docker.txt .

RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir -r requirements_docker.txt

ENV HF_HOME="${BASE_DIR}/hf"
ENV HF_HUB_CACHE="${BASE_DIR}/hf"

COPY handler.py .

# RUN ["chmod", "+x", "/app/character_inference/start_inference.sh"]
CMD ["python", "-u", "handler.py"]