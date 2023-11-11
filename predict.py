import os
from typing import List
import torch
from utils import SobelOperator
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
)
from compel import Compel
from PIL import Image
import numpy as np
from inputparams import InputParams
import os



from cog import BasePredictor, Input, Path


MODEL_CACHE = "weights_cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # check if MPS is available OSX only M1/M2/M3 chips
        device = "cuda:0"
        # change to torch.float16 to save GPU memory
        torch_dtype = torch.float16

        self.controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch_dtype
        ).to(device)

        self.canny_torch = SobelOperator(device=device)

        model_id = "Linaqruf/anything-v3.0"
        lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            controlnet=self.controlnet_canny
        )

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype).to(device)
        self.pipe.unet.to(memory_format=torch.channels_last)

        self.pipe.load_lora_weights(lcm_lora_id)
        self.pipe.load_lora_weights("/root", weight_name="pixel_portraits.safetensors", adapter_name="pixell")
        self.pipe.set_adapters(["lora", "pixell"], adapter_weights=[1.0, 1.0])

        self.pipe.to(device=device, dtype=torch.float16)

        # self.compel_proc = Compel(
        #     tokenizer=self.pipe.tokenizer,
        #     text_encoder=self.pipe.text_encoder,
        #     truncate_long_prompts=False,
        # )

    def predict(
        self,
        image: Path = Input(
            description="Input image",
            default="",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="A person, pixel art",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="bad quality, low quality",
        ),
        strength: float = Input(
            description="Denoising strength", ge=0.0, le=1.0, default=0.5
        ),
        guidance_scale: float = Input(
            description="CFG scale for guidance",
            ge=0.0, le=10.0, default=8.0
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", ge=1, le=50, default=10
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=42
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        input_image = Image.open(str(image))

        params = params = InputParams(
            prompt=prompt,
            negative_prompt = negative_prompt,
            guidance_scale = guidance_scale, 
            steps = num_inference_steps,
            strength = strength, 
            seed = seed,
            height = input_image.size[0],
            width = input_image.size[1]
        )
        # prompt_embeds = self.compel_proc(prompt)

        generator = torch.manual_seed(params.seed)

        control_image = self.canny_torch(
            input_image, params.canny_low_threshold, params.canny_high_threshold
        )
        results = self.pipe(
            control_image=control_image,
            # prompt_embeds=prompt_embeds,
            prompt = prompt,
            generator=generator,
            image=input_image,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
        )
        result_image = results.images[0]

        return result_image
