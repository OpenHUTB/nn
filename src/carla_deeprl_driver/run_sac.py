from source.sac_trainer import Trainer
from source.sac import config, device
from source.carlaenv import CarlaEnv
from tqdm import tqdm
from source.utility import setup_seed

if __name__ == '__main__':
    print("*" * 20, flush=True)
    print(f"use device {device}", flush=True)
    print("*" * 20, flush=True)
    print("Setting up seed...", flush=True)
    setup_seed(20)
    print("Creating environment...", flush=True)
    env = CarlaEnv()
    print("Reloading world...", flush=True)
    env.client.reload_world(False)
    print("Creating trainer...", flush=True)
    trainer = Trainer(env)
    print("Starting training loop...", flush=True)
    for epoch_i in tqdm(range(config['epoch']), desc="Epoch"):
        trainer.train(epoch_i)
            
    print("Training completed!", flush=True)
    trainer.env.exit_env()
