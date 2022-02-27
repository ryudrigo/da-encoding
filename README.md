# da-encoding

  Repo with code for upcoming paper proposal.

## Installation
  Simply run 
  
    pip -r requirements.txt
    
## Usage
  First, run train_resnet18_with_aug.py with desired augmentations (e.g. uncomment the ColorJitter augmentation).
  The checkpoint will be generated in a new folder called lightning_logs. Copy the checkpoint path to load_backbone in ranking_and_backbone.py,
  then choose the desired augmentation (e.g. generate_brighted_imgs) and run that file. Checkpoint will again be in lightning_logs. Copy it to layer_relevance.py and 
  it will print the results.
  
  A more automatized workflow might be avaiable once the paper is finished.
