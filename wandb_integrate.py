#run this file in wandb_env >> requirement.txt

import wandb

wandb.init(
    project='fashion_mnist',
    name='plot sample images from each class',
    config={})
