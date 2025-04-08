from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from platform import system
import shutil
import os


if TYPE_CHECKING:
    from flux_pipeline import FluxPipeline

if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1

STYLES_FOLDER = "/lora_styles"


class AppState:
    model: "FluxPipeline"


class FastAPIApp(FastAPI):
    state: AppState


class LoraArgs(BaseModel):
    scale: Optional[float] = 1.0
    path: Optional[str] = None
    name: Optional[str] = None
    action: Optional[Literal["load", "unload"]] = "load"


class LoraLoadResponse(BaseModel):
    status: Literal["success", "error"]
    message: Optional[str] = None


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
    strength: Optional[float] = 1.0
    init_image: Optional[str] = None
    lora_personal: Optional[str] = None
    lora_styles: Optional[list[LoraStyle]] = None


app = FastAPIApp()


@app.post("/generate")
def generate(args: GenerateArgs):
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
            - `strength`: strength for image generation, 0.0 - 1.0.
                Represents the percent of diffusion steps to run,
                setting the init_image as the noised latent at the
                given number of steps.
            - `init_image`: Base64 encoded image or path to image to use as the init image.

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    """
    loaded_loras = []
    if not args.lora_personal is None:
        app.state.model.load_lora(args.lora_personal, 1.0, "user")
        loaded_loras.append("user")
    if not args.lora_styles is None:
        for style in args.lora_styles:
            path = style.path
            basename = os.path.basename(path)
            local_path = os.path.join(STYLES_FOLDER, basename)
            if not os.path.exists(local_path):  # download and load from local
                shutil.copy(path, local_path)

            if style.name is None:
                style.name = basename.split(".safet")[0]
            app.state.model.load_lora(local_path, style.scale, style.name)
            loaded_loras.append(style.name)

    vars = args.model_dump()
    vars.pop("lora_personal")
    vars.pop("lora_styles")
    result = app.state.model.generate(**vars)
    for name in loaded_loras:
        app.state.model.unload_lora(name)
    return StreamingResponse(result, media_type="image/jpeg")


@app.post("/lora", response_model=LoraLoadResponse)
def lora_action(args: LoraArgs):
    """
    Loads or unloads a LoRA checkpoint into / from the Flux flow transformer.

    Args:
        args (LoraArgs): Arguments for the LoRA action:
            - `scale`: The scaling factor for the LoRA weights.
            - `path`: The path to the LoRA checkpoint.
            - `name`: The name of the LoRA checkpoint.
            - `action`: The action to perform, either "load" or "unload".

    Returns:
        LoraLoadResponse: The status of the LoRA action.
    """
    try:
        if args.action == "load":
            app.state.model.load_lora(args.path, args.scale, args.name)
        elif args.action == "unload":
            app.state.model.unload_lora(args.name if args.name else args.path)
        else:
            return JSONResponse(content={"status": "error",
                                         "message": f"Invalid action, expected 'load' or 'unload', got {args.action}",},
                                status_code=400,)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    return JSONResponse(status_code=200, content={"status": "success"})
