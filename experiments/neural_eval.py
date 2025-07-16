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
        raise FileNotFoundError(
            f"Model config file {model_config_path} does not exist."
        )
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(
            f"Dataset config file {dataset_config_path} does not exist."
        )

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    dataset = load_dataset(dataset_config)
    dataset.transform = resolve_transform(model_config)
    print(f"Loaded validation set with {len(dataset)} samples.")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    print(f"DataLoader created with batch size {args.batch_size}.")

    # load pipeline
    pipeline = load_pipeline(model_config)
    pipe = pipeline["pipe"].to("cuda")
    vae = pipeline["vae"].to("cuda")
    print("Pipeline loaded successfully.")

    for i, (image, _) in enumerate(tqdm(loader)):
        for rep in range(args.n_reps):
            print(f"Processing batch {i}, repetition {rep + 1}/{args.n_reps}")
            with torch.no_grad():
                image = image.to("cuda").to(dtype=torch.float16)

                latent = vae.encode(image).latent_dist.sample()
                print(vae.config.scaling_factor)
                latent = latent * vae.config.scaling_factor

                # forward process
                noise = torch.randn_like(latent)
                t_max = torch.tensor([args.t_max], device=latent.device)
                noised_latent = pipe.scheduler.add_noise(latent, noise, t_max)

                # backward process
                print("Batch ID:", i)
                denoised = pipe.truncated(
                    noised_latents=noised_latent,
                    start_t=t_max,
                    prompt=[model_config["default-prompt"]] * noised_latent.shape[0],
                    batch_id=i,
                    rep_id=rep,
                    model_name=model_config["model"],
                    dataset_name=dataset_config["dataset"],
                )

                for j, im in enumerate(denoised.images):
                    save_path = f"outputs/images/{dataset_config['dataset']}/denoised_{i}_{j}_rep_{rep}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    im.save(save_path)

            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Noising Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model config filename, sans .yaml (e.g., sdxl, ltxv, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset config filename, sans .yaml (e.g., imagenet)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for processing"
    )
    parser.add_argument(
        "--t_max", type=int, default=501, help="Maximum timestep for noising"
    )
    parser.add_argument(
        "--n_reps", type=int, default=1, help="Number of repetitions for each batch"
    )

    args = parser.parse_args()
    main(args)
