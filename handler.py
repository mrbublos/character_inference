import os
import io
from typing import Optional
import runpod 
import torch
import numpy as np
from pydantic import BaseModel, Field
import shutil
from diffusers.image_processor import VaeImageProcessor
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL

import base64

# Constants

MAX_RAND = 2**32 - 1
MAX_GPU_MEMORY = int(torch.cuda.mem_get_info(0)[1] / 1024 ** 2 / 1000)
BASE_DIR = os.getenv("BASE_DIR")

STYLES_FOLDER = f"{BASE_DIR}/lora_styles"
USER_MODELS = f"{BASE_DIR}/user_models"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
os.environ["HF_HOME"] = f"{BASE_DIR}/hf"
os.environ["HF_HUB_CACHE"] = f"{BASE_DIR}/hf"

logger = runpod.RunPodLogger()

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
    lora_personal: Optional[bool] = None
    lora_styles: Optional[list[LoraStyle]] = None
    user_id: str

def flush():
    """Clear CUDA memory cache"""
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

class FluxGenerator:
    def __init__(self):
        self.setup_environment()
        self.load_models()
        os.makedirs(STYLES_FOLDER, exist_ok=True)

    def setup_environment(self):
        """Configure PyTorch and CUDA settings"""
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 20
        torch.set_float32_matmul_precision("high")

    def load_models(self):
        """Load all required models"""
        logger.info("Loading models...")
        dtype = torch.float16
        
        max_memory = {k: "0GB" for k in range(1)}
        max_memory[1] = str(MAX_GPU_MEMORY) + "GB"
        max_memory["cpu"] = "0GB"

        # Load encoder
        self.encoder = FluxPipeline.from_pretrained(
            "black-forest-labs/flux.1-dev",
            transformer=None,
            vae=None,
            torch_dtype=dtype,
            max_memory=max_memory,
        )
        self.encoder.to("cuda")

        # Load transformer with quantization
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/flux.1-dev",
            subfolder="transformer",
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=dtype
        )

        # Initialize pipeline with transformer
        self.model = FluxPipeline.from_pretrained(
            "black-forest-labs/flux.1-dev",
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            torch_dtype=dtype
        )
        self.model.to("cuda")

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/flux.1-dev",
            subfolder="vae",
            torch_dtype=dtype,
            max_memory=max_memory,
        )
        self.vae.to("cuda")

        # Warmup
        logger.info("Performing warmup inference...")
        self.generate({
            "prompt": "Warmup inference",
            "width": 1024,
            "height": 1024,
            "num_steps": 4,
            "guidance": 3.5,
            "seed": 10
        })

    def process_lora_styles(self, args: dict) -> dict:
        """Process and validate LoRA style paths"""
        if args.get("lora_styles"):
            for i, style in enumerate(args["lora_styles"]):
                path = BASE_DIR + style["path"]
                if not os.path.exists(path):
                    args["lora_styles"][i]["path"] = ""
                    continue
                
                basename = os.path.basename(path)
                logger.info(f"Using style {basename}")

                if not args["lora_styles"][i].get("name"):
                    args["lora_styles"][i]["name"] = basename.split(".safet")[0]
        return args

    def generate(self, args: dict) -> bytes:
        """Generate image from input parameters"""
        try:
            # Encode prompt
            with torch.inference_mode():
                prompt_embeds, pooled_prompt_embeds, _ = self.encoder.encode_prompt(
                    args["prompt"],
                    prompt_2=None,
                    max_sequence_length=512,
                    num_images_per_prompt=1
                )

            # Handle LoRA loading
            lora_names = []
            lora_scales = []
            
            if args.get("lora_personal"):
                logger.info("Using personal style")
                self.model.load_lora_weights(f"{USER_MODELS}/{args['user_id']}/{args['user_id']}.safetensors", adapter_name="user")
                lora_names.append("user")
                lora_scales.append(1.0)

            if args.get("lora_styles"):
                for style in args["lora_styles"]:
                    if not style["path"]:
                        continue
                    self.model.load_lora_weights(style["path"], adapter_name=style["name"])
                    lora_names.append(style["name"])
                    lora_scales.append(style["scale"])

            if lora_names:
                self.model.set_adapters(lora_names, adapter_weights=lora_scales)

            # Generate latents
            with torch.inference_mode():
                latents = self.model(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=args["num_steps"],
                    guidance_scale=args["guidance"],
                    height=args["height"],
                    width=args["width"],
                    output_type="latent"
                ).images

            # Unload LoRAs if used
            if lora_names:
                self.model.unload_lora_weights(reset_to_overwritten_params=True)

            flush()

            logger.info("Decondng image")
            # Decode image
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            
            with torch.inference_mode():
                latents = FluxPipeline._unpack_latents(latents, args["height"], args["width"], vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = image_processor.postprocess(image, output_type="pil")[0]

            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=95)
            return img_byte_arr.getvalue()

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")

# Initialize the generator
generator = FluxGenerator()

def handler(event):
    """RunPod handler function"""
    try:
        # Validate input
        if not event.get("input"):
            raise ValueError("No input data provided")
        
        # Parse and validate arguments
        args = GenerateArgs(**event["input"])
        input_dict = args.model_dump()
        
        # Process LoRA styles
        input_dict = generator.process_lora_styles(input_dict)
        
        # Generate image
        image_bytes = generator.generate(input_dict)
        
        return {
            "output": {
                "image": base64.b64encode(image_bytes),
                "user_id": args['user_id'],
            }
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
