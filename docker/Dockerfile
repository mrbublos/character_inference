FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1

# RUN git clone https://github.com/Otnielush/character_inference.git
WORKDIR /app/character_inference
COPY . .
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir -r requirements_docker.txt
RUN apt-get install -y tmux nvtop htop


ENV TORCH_CUDA_ARCH_LIST="Turing"
ENV PATH="/opt/conda/bin:/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.2/compat/:/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="video,compute,utility"
ENV CUDA_HOME='/usr/local/cuda'

# WORKDIR /repos
# RUN git clone https://github.com/aredden/torch-cublas-hgemm.git
# WORKDIR /repos/torch-cublas-hgemm
# RUN python3 -m pip install --no-build-isolation -U -v .


ENV HF_HOME="/workspace/hf"
ENV HF_HUB_CACHE="/workspace/hf"
ENV NUM_GPUS=1

# Important paths
# dir for saving Lora styles
WORKDIR /lora_styles

EXPOSE 8088

WORKDIR /app/character_inference
# RUN ["chmod", "+x", "/app/character_inference/start_inference.sh"]
CMD ["python3 inference_hf.py --port 8088 --host 0.0.0.0 --no_offload_embd"]

# run container important params:
    # --gpus=all  - give access to gpus
    # -e NUM_GPUS=_number of GPUs, default 1
    # -p _port_:8088  - map port
    # -v /workspace/hf  - dir for models

