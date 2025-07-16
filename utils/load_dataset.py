import torch
from torchvision import datasets, transforms

from utils.datasets import ArcaroDataset


def load_dataset(config):
    """
    Load the dataset specified in the config file.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        Dataset: Loaded dataset.
    """

    if config["dataset"] == "imagenet":
        # train_set = datasets.ImageNet(root=config['paths']['root'], split='train')
        val_set = datasets.ImageFolder(root=config["paths"]["val"])
        return val_set
    if config["dataset"] == "arcaro":
        dataset = ArcaroDataset(
            root=config["paths"]["root"],
            monkey=config.get("monkey", "Red"),
        )
        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")


def resolve_transform(model_config):
    """
    Resolve the transformation to be applied to the dataset based on the model configuration.

    Args:
        model_config (dict): Configuration dictionary for the model.

    Returns:
        transforms.Compose: Composed transformations.
    """

    resize = transforms.Resize(
        model_config["image-size"]["height"],
        transforms.InterpolationMode.BICUBIC,
    )
    crop = (
        transforms.CenterCrop(
            (model_config["image-size"]["height"], model_config["image-size"]["width"])
        )
        if model_config.get("center-crop", False)
        else transforms.Lambda(lambda x: x)
    )
    # assume imagenet for now
    normalize = transforms.Normalize(
        mean=model_config.get("normalize", {}).get("mean", [0.5, 0.5, 0.5]),
        std=model_config.get("normalize", {}).get("std", [0.5, 0.5, 0.5]),
    )

    transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return transform
