import torch
from diffusers import AutoencoderKL
from custom_pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline


def load_pipeline(config):
    if config['model'] == 'sdxl':
        try:
            caching_layers = config['probing']['layers']
        except:
            caching_layers = None

        print(caching_layers)



        print("Loading Stable Diffusion XL pipeline...")
        sdxl = StableDiffusionXLPipeline.from_pretrained(
            config['hf-name'],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16',
            caching_layers=caching_layers,
            caching_timestep_interval=10
        )
        # need to use separate VAE bc of bug in fp16 SDXL VAE
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        return {'pipe': sdxl, 'vae': vae}
