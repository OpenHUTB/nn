from semantic.carla_controller.labels import SEMANTIC_CATEGORIES
from semantic.unet.model import get_unet
from semantic.unet.train import train_unet

from datetime import datetime


img_size = (2556,256)#(512, 512)

model = get_unet(img_size, len(SEMANTIC_CATEGORIES))
model = train_unet(
    model,
    50,
    8,
    img_size,
    test_size=0.25,
    dataset_folder="./dataset",
    # load_from_checkpoint="./checkpoints/unet.h5"
)

model.save(f"./unet_model_{datetime.utcnow()}")