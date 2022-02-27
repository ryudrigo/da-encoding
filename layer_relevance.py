import torch
from pytorch_lightning import Trainer
from ranking_and_backbone import Ranking
from torch.utils.data import DataLoader
from train_resnet18_with_aug import Preprocess
from torchvision.datasets import STL10
import os
from ranking_and_backbone import my_collate

if __name__ == "__main__":

    preprocess = Preprocess()

    #contrast, con/sat augmentation (maybe!): version 125 to 131, inclusive
    #saturation, full (con/sat/bri/hue) augmentation: version 132 to 138, inclusive
    #saturation, no augmentation: version 141 to 147, inclusive
    #brightness, no augmentation: version 148 to , inclusive
    checkpoint = torch.load('lightning_logs/version_148/checkpoints/epoch=2.ckpt')
    ranking = Ranking()
    ranking.load_state_dict(checkpoint['state_dict'], strict=False)

    dataset = STL10(os.getcwd(), split='test', download=True, transform=preprocess)
    data_loader = DataLoader(dataset, batch_size=64,collate_fn=my_collate)

    layers_multiplicity=[1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8]

    a =ranking.linear.weight[0][0:3904]
    b = ranking.linear.weight[0][3904:7808]

    layers_a=[]
    layers_b=[]
    with torch.no_grad():
        i=0
        j=0
        while (i<61):
            x = a[i*64:(i+layers_multiplicity[j])*64]
            i=i+layers_multiplicity[j]
            j+=1
            layers_a.append(x)
        i=0
        j=0
        while (i<61):
            x = b[i*64:(i+layers_multiplicity[j])*64]
            i=i+layers_multiplicity[j]
            j+=1
            layers_b.append(x)
     
    all_acts = []
    for i, batch in enumerate(iter(data_loader)):
        ranking.backbone(batch[0])
        act, names = ranking.get_activations()
        all_acts.append(act)
    all_acts= torch.stack(all_acts)
    i=0
    j=0
    layers_product_a=[]
    layers_product_b=[]
    with torch.no_grad():
        while (i<61):
            x = all_acts[:,:,0,i*64:(i+layers_multiplicity[j])*64]
            x = x.reshape(x.shape[0]*x.shape[1], -1)
            x=x.std(dim=(0))
            z_a=x*layers_a[j]
            z_b=x*layers_b[j]
            layers_product_a.append(sum(z_a)/len(z_a))
            layers_product_b.append(sum(z_b)/len(z_b))
            i=i+layers_multiplicity[j]
            j+=1

    print ('without product:')
    for auau in layers_a:
        print(float((sum(auau)/len(auau))*1000))
    print ('\n\n')
    for auau in layers_b:
        print(float((sum(auau)/len(auau))*1000))


    layer_mean_impact_a = layers_product_a
    layer_mean_impact_b = layers_product_b
    print ('with product:')
    for auau in layer_mean_impact_a:
        print(float(auau*1000))
    print ('\n\n')
    for auau in layer_mean_impact_b:
        print(float(auau*1000))
