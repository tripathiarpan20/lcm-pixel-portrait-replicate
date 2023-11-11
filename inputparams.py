from pydantic import BaseModel
class InputParams(BaseModel):
    seed: int = 2159232
    prompt: str
    negative_prompt: str = "bad quality, low quality"
    guidance_scale: float = 8.0
    strength: float = 0.5
    steps: int = 4
    lcm_steps: int = 50
    width: int = 512
    height: int = 512
    controlnet_scale: float = 0.8
    controlnet_start: float = 0.0
    controlnet_end: float = 1.0
    canny_low_threshold: float = 0.31
    canny_high_threshold: float = 0.78
    debug_canny: bool = False