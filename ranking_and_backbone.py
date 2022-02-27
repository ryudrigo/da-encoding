import pytorch_lightning as pl
import math
import torchvision.models as models
import torchvision.transforms as T
from train_resnet18_with_aug import ResNetSystem
from train_resnet18_with_aug import DataAugmentation
from train_resnet18_with_aug import Preprocess
from PIL import Image
from torch import Tensor
import os
from dotmap import DotMap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as torch_data_utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import random
import os
from torchvision.datasets import STL10



def my_collate(batch):
    # item: a tuple of (img, label)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]
    
class Ranking (pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size=128
        self.backbone = self.load_backbone()
        self.hooks = {}
        for name, module in self.backbone.named_modules():
            self.hooks[name] = module.register_forward_hook(self.save_activation(name))
        self.linear = torch.nn.Linear(61*64*2, 1)
        self.flatten = torch.nn.Flatten()
        self.writer = SummaryWriter()
        self.many_activations=[]
        self.first_layer_name = '0'
        self.preprocess= Preprocess()
    
    def save_activation(self, name):
        def activations_hook(module, act_in, act_out):
            if self.first_layer_name == name:
                self.many_activations.append ({})
            self.many_activations[-1][name]=act_out.detach()
        return activations_hook
        
    def load_backbone(self):

        #no data augmentation: checkpoint = torch.load("lightning_logs/version_124/checkpoints/epoch=68.ckpt")
        #with Contrast/Saturation data augmentation checkpoint = torch.load("lightning_logs/version_84/checkpoints/epoch=208.ckpt")
        #with Contrast/Saturation/Brightness/Hue data augmentation (more epochs) checkpoint = torch.load("lightning_logs/version_86/checkpoints/epoch=409.ckpt")
        checkpoint = torch.load("lightning_logs/version_124/checkpoints/epoch=68.ckpt")
        backbone = models.resnet18(pretrained=False)
        backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        feature_extractor = nn.Sequential (*layers)
        backbone_model = feature_extractor
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone_model
        
    def get_activations(self):
        big_list_of_acts=[]
        list_of_names=[self.first_layer_name] #this is for pairing up activations and their layers.
        for activations in self.many_activations:
            list_of_acts = [activations[self.first_layer_name].mean(dim=(-2, -1))]
            for ra, name in enumerate(list(activations.keys())):
                if 'conv' in name:
                    num_of_feat_maps = activations[name].shape[-3]
                    act = activations[name].mean(dim=(-2, -1))
                    act=torch.chunk(act, num_of_feat_maps//64, dim=-1)
                    list_of_acts+=act
                    if len(list_of_names)<61: #there are 61 layer chunks in total
                        for _ in range (num_of_feat_maps//64):
                            list_of_names.append(name)
            list_of_acts = torch.cat(list_of_acts, dim=1)
            big_list_of_acts.append (list_of_acts)
        answer = torch.stack(big_list_of_acts)
        #the following reorganizes the tensor so that first dimension is batch_size, second dimension is the pair of input examples and third dimension is the flattened backbone activation data
        #because shape of answer was (2, batch_size, 61*64)
        #so now it will be (batch_size, 2, 61*64) 
        #2 comes from the two images, each with an augmentation
        #61 comes from 61 convolutional layers (some of the original 17 were broken in chunks)
        #64 comes from the number of channels
        answer= answer.permute((1,0, 2))
        self.many_activations=[]
        return answer, list_of_names #list_of_names can be used when loding model from another script
    
    def generate_saturated_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while (abs(aug_value1-aug_value2)<0.2):
            aug_value1 = random.uniform(0.5, 1.5)
            aug_value2 = random.uniform(0.5, 1.5)
        img1 = TF.adjust_saturation(img, aug_value1)
        img2 = TF.adjust_saturation(img, aug_value2)            
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if aug_value1>aug_value2:
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value
        
    def generate_hued_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while (abs(aug_value1-aug_value2)<0.1):
            aug_value1 = random.uniform(-0.4, 0.4)
            aug_value2 = random.uniform(-0.4, 0.4)
        img1 = TF.adjust_hue(img, aug_value1)
        img2 = TF.adjust_hue(img, aug_value2)            
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if aug_value1>aug_value2:
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value
    
    def generate_contrasted_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while (abs(aug_value1-aug_value2)<0.2):
            aug_value1 = random.uniform(0.5, 1.5)
            aug_value2 = random.uniform(0.5, 1.5)
        img1 = TF.adjust_contrast(img, aug_value1)
        img2 = TF.adjust_contrast(img, aug_value2)            
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if aug_value1>aug_value2:
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value
    
    def generate_brighted_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while (abs(aug_value1-aug_value2)<0.2):
            aug_value1 = random.uniform(0.5, 1.5)
            aug_value2 = random.uniform(0.5, 1.5)
        img1 = TF.adjust_brightness(img, aug_value1)
        img2 = TF.adjust_brightness(img, aug_value2)            
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if aug_value1>aug_value2:
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value
    
    def generate_rotated_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while (abs(aug_value1-aug_value2)<5):
            aug_value1 = random.uniform(-60, 60)
            aug_value2 = random.uniform(-60, 60)
        img1 = TF.rotate(img, aug_value1, expand=False)
        img2 = TF.rotate(img, aug_value2, expand=False)  
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if aug_value1>aug_value2:
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value
        
    def generate_abs_rotated_imgs(self, img):
        aug_value1=0
        aug_value2=0
        while abs(abs(aug_value1)-abs(aug_value2))<5:
            aug_value1 = random.uniform(-60, 60)
            aug_value2 = random.uniform(-60, 60)
        img1 = TF.rotate(img, aug_value1, expand=False)
        img2 = TF.rotate(img, aug_value2, expand=False)  
        img1 = TF.resize(img1, 180)
        img2 = TF.resize(img2, 180)
        img1 = TF.center_crop(img1, 128)
        img2 = TF.center_crop(img2, 128)
        if abs(aug_value1)>abs(aug_value2):
            y_value = 1.
        else:
            y_value = 0.
        return img1, img2, y_value    
   
    
    def generate_backbone_activations(self, imgs):
        x=[]
        y=[]
        imgs1=[]
        imgs2=[]
        for img in imgs:          
            img1, img2, y_value = self.generate_brighted_imgs(img)
            imgs1.append(img1)
            imgs2.append(img2)
            y.append(y_value)
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        self.backbone(imgs1)
        self.backbone(imgs2)
        act, _ = self.get_activations() #which layer can come from act1 or act2, they're equal
        y=torch.as_tensor(y,device=torch.device('cuda'))
        y = torch.unsqueeze(y, dim=1)
        x=act
        return x, y, imgs1, imgs2 #returns images for debugging purposes
        
    def forward(self, backbone_activations):
        backbone_activations = self.flatten(backbone_activations)
        y_hat = torch.sigmoid(self.linear(backbone_activations))
        return y_hat
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x, y, _, _ = self.generate_backbone_activations(x)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/train', loss, self.current_epoch*self.batch_size+batch_idx)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x, y, _, _ = self.generate_backbone_activations(x)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/val', loss, self.current_epoch*self.batch_size+batch_idx)
        
        pred = [math.floor(x+0.5) for x in y_hat]
        correct_pred = sum([float(a==b) for a, b in zip (pred, y)])
        acc = correct_pred / (len(y) * 1.0)
        self.writer.add_scalar('Val/acc', acc, self.current_epoch*self.batch_size+batch_idx)
        self.writer.add_scalar('Val/mean_pred', sum(pred)/len(pred), self.current_epoch*self.batch_size+batch_idx)
        return {"loss": loss}
    
    def train_dataloader(self):
        dataset = STL10(os.path.join(os.getcwd(), 'data'), split='train', download=True, transform=self.preprocess)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
   
    def val_dataloader(self):
        dataset = STL10(os.path.join(os.getcwd(), 'data'), split='test', download=True, transform=self.preprocess)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
    1
if __name__ == "__main__":
    trainer = pl.Trainer(gpus='0', max_epochs=3)
    model = Ranking()
    trainer.fit (model)