from PIL import Image
from custom_pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                  torch_dtype=torch.float16,
                                                  use_safetensors=True,
                                                  variant="fp16",
                                                  caching_layers=["down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_q"],caching_timestep_interval=10)
print(pipe.unet)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("outputs/astronaut_riding_horse.png")
