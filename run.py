from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

from img_model import ImageClassifier


class TVDataset(Dataset):

    '''
    pls dont do this 😅
    '''

    def __init__(self, ds, feature_extractor):
        self.ds = ds
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        inputs = self.feature_extractor(x, return_tensors='pt')['pixel_values'].squeeze(0)
        return inputs, y
    
    def __len__(self):
        return len(self.ds)


if __name__ == '__main__':

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 10
    batch_size = 16
    num_workers = 2
    max_epochs = 4

    train_loader = DataLoader(
        TVDataset(CIFAR10('./', download=True), ViTFeatureExtractor.from_pretrained(model_name_or_path)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    model = ImageClassifier(model_name_or_path)
    # HACK - put this somewhere else
    model.total_steps = (
        (len(train_loader.dataset) // (batch_size))
        // 1
        * float(max_epochs)
    )
    pixel_values, labels = next(iter(train_loader))
    trainer = pl.Trainer(gpus=1, max_epochs=4, precision=16)
    trainer.fit(model, train_loader)
