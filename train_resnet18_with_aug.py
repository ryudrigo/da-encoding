import os
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline, Resize,Normalize
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from torch.utils.tensorboard import SummaryWriter   

class DataAugmentation(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.75),
            RandomThinPlateSpline(p=0.75),
            #ColorJitter(contrast=.7, saturation=.7, brightness =.7, hue =.7)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out

class Preprocess(nn.Module):

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        #do we normalize here?
        return x_out.float() / 255.0


class ResNetSystem(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)

        self.preprocess = Preprocess()  # per sample transforms

        self.transform = DataAugmentation()  # per batch augmentation_kornia

        self.accuracy = torchmetrics.Accuracy()
        
        self.pil_transform = T.ToPILImage()
        
        self.writer = SummaryWriter()
        
        self.batch_size=64

    def forward(self, x):
        return F.softmax(self.model(x))

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def show_batch(self):
        
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply transforms
        
        for i, img in enumerate(imgs_aug):
            img=self.pil_transform(img)
            img.save('temp3/{}.jpg'.format(i))
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_aug = self.transform(x)  # => we perform GPU/Batched data augmentation
        y_hat = self(x_aug)
        loss = self.compute_loss(y_hat, y)
        self.writer.add_scalar("Train/loss", loss, self.current_epoch*self.batch_size+batch_idx)
        self.writer.add_scalar("Train/acc", self.accuracy(y_hat, y), self.current_epoch*self.batch_size+batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.writer.add_scalar("Val/loss", loss, self.current_epoch*self.batch_size+batch_idx)
        self.writer.add_scalar("Val/acc", self.accuracy(y_hat, y), self.current_epoch*self.batch_size+batch_idx)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

    def prepare_data(self):
        
        STL10(os.path.join(os.getcwd(), 'data'), split='train', download=True,  transform=self.preprocess)
        STL10(os.path.join(os.getcwd(), 'data'), split='test', download=True,  transform=self.preprocess)
        

    def train_dataloader(self):
        dataset = STL10(os.path.join(os.getcwd(), 'data'), split='train', download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=6)
        return loader

    def val_dataloader(self):
        dataset = STL10(os.path.join(os.getcwd(), 'data'), split='test', download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=6)
        return loader
        
if __name__ == "__main__":
    rng = torch.manual_seed(28)
    model = ResNetSystem()
    model.show_batch()
    trainer = Trainer(gpus='0', max_epochs=500)    
    #trainer = Trainer(gpus='0', max_epochs=10000, resume_from_checkpoint = "lightning_logs/version_77/checkpoints/epoch=158.ckpt")
    trainer.fit(model)
    