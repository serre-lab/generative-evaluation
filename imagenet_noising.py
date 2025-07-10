# TODO
# 1. Load the ImageNet dataset from the config file.
# 2. Load the model specified in the config file.
# 3. For each batch, apply the noising function up to t_max.
# 4. Run the backward process with the model and save activations at specified timesteps.
 
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from diffusers import AutoencoderKL

from utils.load_pipeline import load_pipeline
from utils.load_dataset import load_dataset, resolve_transform

def main(args):
    # load configs
    model_config_path = f"configs/model_configs/{args.model}.yaml"
    dataset_config_path = f"configs/data_configs/{args.dataset}.yaml"
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file {model_config_path} does not exist.")
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"Dataset config file {dataset_config_path} does not exist.")
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    val_set = load_dataset(dataset_config)
    val_set.transform = resolve_transform(model_config)
    print(f"Loaded validation set with {len(val_set)} samples.")
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"DataLoader created with batch size {args.batch_size}.")

    # load pipeline
    pipeline = load_pipeline(model_config)
    pipe = pipeline['pipe'].to("cuda")
    vae = pipeline['vae'].to("cuda")
    print("Pipeline loaded successfully.")
    
    for image, _ in tqdm(val_loader):
        with torch.no_grad():
            image = image.to("cuda").to(dtype=torch.float16)
            print("Image shape:", image.shape, image.dtype, image.min(), image.max(), image.mean())

            latent = vae.encode(image).latent_dist.sample()
            print(vae.config.scaling_factor)
            latent = latent * vae.config.scaling_factor

            # forward process
            noise = torch.randn_like(latent)
            t_max = torch.tensor([300], device=latent.device)
            noised_latent = pipe.scheduler.add_noise(latent, noise, t_max)

            # backward process
            denoised = pipe.truncated(
                noised_latent=noised_latent,
                t=t_max,
                prompt=model_config['default-prompt']
            )

            decoded_image = vae.decode(noised_latent / vae.config.scaling_factor).sample

            # plt.imshow(decoded_image[0].to(dtype=torch.float32).cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            # plt.axis('off')
            # plt.savefig("decoded_image.png", bbox_inches='tight', pad_inches=0)

            # forward process
            # noise = torch.randn_like(latent)
            # t_max = torch.tensor([1], device=latent.device)
            # # noised_latent = pipe.scheduler.add_noise(latent, noise, t_max)
            # noised_latent = latent
            
            # # just decode for the moment
            # decoded_image = pipe.vae.decode(noised_latent / pipe.vae.config.scaling_factor).sample
            # print(decoded_image.shape, decoded_image.dtype, decoded_image.min(), decoded_image.max())
            break

        torch.cuda.empty_cache()

        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageNet Noising Script")
    parser.add_argument('--model', type=str, required=True, help='Model config filename, sans .yaml (e.g., sdxl, ltxv, etc.)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset config filename, sans .yaml (e.g., imagenet)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    
    args = parser.parse_args()
    main(args)
