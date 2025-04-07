import argparse
import time
from typing import Any, List, Optional
import random
import os
from pydantic import BaseModel, Field
from platform import system
import shutil
import io

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np


# from torch.multiprocessing import set_start_method, Queue, Process, Pipe, current_process
from multiprocessing import set_start_method, Queue, Process, Pipe, current_process
try:
     set_start_method('spawn')
except RuntimeError:
    pass




if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1

# Configuration
NUM_GPUS = int(os.environ.get("NUM_GPUS", 1))  # Number of simulated GPUs
TASK_QUEUE = Queue()
GPU_PROCESSES: List[Process] = []
STYLES_FOLDER = "/lora_styles"



# torch.multiprocessing.set_start_method('spawn')


class LoraStyle(BaseModel):
    path: str
    scale: float = Field(default=1.0)
    name: Optional[str] = None


class GenerateArgs(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=1024)
    height: Optional[int] = Field(default=720)
    num_steps: Optional[int] = Field(default=28)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND)
    lora_personal: Optional[LoraStyle] = None
    lora_styles: Optional[list[LoraStyle]] = None



app = FastAPI()




def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    # parser.add_argument("-c", "--config-path", type=str, help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments")
    parser.add_argument("-p", "--port", type=int, default=8088, help="Port to run the server on")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host to run the server on")

    return parser.parse_args()




def process_data_on_gpu(model, vars: dict[str, Any], pipe_in: Pipe):
    # loading loras
    lora_names = []
    lora_scales = []
    if not vars['lora_personal'] is None:
        model.load_lora_weights(vars['lora_personal']['path'], adapter_name='user')
        lora_names.append("user")
        lora_scales.append(1.0)
    if not vars['lora_styles'] is None:
        for style in vars['lora_styles']:
            model.load_lora_weights(style['path'], adapter_name=style['name'])
            lora_names.append(style['name'])
            lora_scales.append(style['scale'])

    model.set_adapters(lora_names, adapter_weights=lora_scales)

    vars['num_inference_steps'] = vars['num_steps']
    vars['guidance_scale'] = vars['guidance']
    del(vars['guidance'], vars['num_steps'], vars['lora_personal'], vars['lora_styles'], vars['seed'])

    print(f"GPU processing: {vars['prompt'][:40]}")
    image = model(**vars).images[0]
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()
    pipe_in.send(imgByteArr)
    pipe_in.send(None)  # Signal completion

    if len(lora_names) > 0:
        model.unload_lora_weights(reset_to_overwritten_params=True)
    # for name in loaded_loras:
    #     model.unload_lora_weights(name)




def gpu_worker(gpu_id: int, task_queue):
    max_memory = {gpu_id: "24GB"}
    # max_memory = None
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    # torch.cuda.set_device(gpu_id)

    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers import FluxTransformer2DModel, FluxPipeline
    from transformers import T5EncoderModel


    print(f"GPU {gpu_id} loading model. {max_memory = }")
    # start model on GPU
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        max_memory=max_memory,
        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16)  #.to(device)

    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        max_memory=max_memory,
        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16) #.to(device)

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        max_memory=max_memory,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.float16) #, device_map="cuda:"+str(gpu_id))
    pipeline.enable_model_cpu_offload(gpu_id)
    print(f"gpu {gpu_id}; exe device = {pipeline._execution_device}")
    # pipeline._execution_device = gpu_id


    # generation loop
    print(f"GPU {gpu_id} worker processes started.")
    while True:
        task = task_queue.get()
        if task is None:  # Shutdown signal
            return
        vars, pipe_in = task
        # child_conn = pickle.loads(child_conn)
        print(f"GPU worker {current_process().name} received task: {vars['prompt'][:40]}")
        process_data_on_gpu(pipeline, vars, pipe_in)


def output_generator(pipe_out: Pipe):
    """
    Generator function to yield output from the GPU worker via the pipe.
    """
    while True:
        try:
            output = pipe_out.recv()
            if output is None:
                # print("output_generator: None, closing")
                pipe_out.close()
                break
            yield output
        except EOFError:
            # print("output_generator: EOF, closing")
            pipe_out.close()
            break


@app.post("/generate")
async def generate(args: GenerateArgs):
    """
    Generates an image from the Flux flow transformer.

    Args:
        args (GenerateArgs): Arguments for image generation:
            - `prompt`: The prompt used for image generation.
            - `width`: The width of the image.
            - `height`: The height of the image.
            - `num_steps`: The number of steps for the image generation.
            - `guidance`: The guidance for image generation, represents the
                influence of the prompt on the image generation.
            - `seed`: The seed for the image generation.

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    """

    # downloading if need lora styles
    if not args.lora_styles is None:
        for i in range(len(args.lora_styles)):
            path = args.lora_styles[i].path
            basename = os.path.basename(path)
            local_path = os.path.join(STYLES_FOLDER, basename)
            if not os.path.exists(local_path):  # download and load from local
                shutil.copy(path, local_path)

            args.lora_styles[i].path = local_path

            if args.lora_styles[i].name is None:
                args.lora_styles[i].name = basename.split(".safet")[0]


    vars = args.model_dump()
    pipe_out, pipe_in = Pipe(duplex=False)
    TASK_QUEUE.put((vars, pipe_in))  # Put data and the child pipe end in the queue
    # return StreamingResponse(output_generator(pipe_out), media_type="text/plain")
    return StreamingResponse(output_generator(pipe_out), media_type="image/jpeg")



def shutdown_gpus():
    """
    Signals the GPU worker processes to shut down when the FastAPI app stops.
    """
    for _ in range(NUM_GPUS):
        TASK_QUEUE.put(None)
    for process in GPU_PROCESSES:
        process.join()
    print("GPU worker processes stopped.")



# app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_gpus)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(STYLES_FOLDER, exist_ok=True)

    for i in range(NUM_GPUS):
        process = Process(target=gpu_worker, args=(i, TASK_QUEUE,), name=f"GPU-{i}")
        GPU_PROCESSES.append(process)
        process.start()
    print("GPU worker processes launched.")

    uvicorn.run(app, host=args.host, port=args.port)
