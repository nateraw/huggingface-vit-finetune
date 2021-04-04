from transformers import ViTFeatureExtractor, ViTForImageClassification, BatchFeature
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

import torch
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

from img_model import ImageClassifier


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path):
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        transform = []

        if feature_extractor.do_resize:
            transform.append(Resize(feature_extractor.size))

        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)


if __name__ == '__main__':

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 10
    batch_size = 24
    num_workers = 2
    max_epochs = 4

    train_loader = DataLoader( 
        CIFAR10('./', download=True, transform=ViTFeatureExtractorTransforms(model_name_or_path)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )
    val_loader = DataLoader( 
        CIFAR10('./', download=True, train=False, transform=ViTFeatureExtractorTransforms(model_name_or_path)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )
    model = ImageClassifier(model_name_or_path)
    # HACK - put this somewhere else
    model.total_steps = (
        (len(train_loader.dataset) // (batch_size))
        // 1
        * float(max_epochs)
    )
    pixel_values, labels = next(iter(train_loader))
    trainer = pl.Trainer(gpus=1, max_epochs=4, precision=16, limit_train_batches=5)
    trainer.fit(model, train_loader, val_loader)
