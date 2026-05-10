import sys
import traceback

print("=" * 50)
print("Starting CARLA RL Training Debug Script")
print("=" * 50)

try:
    print("\n[1/6] Importing basic modules...")
    import numpy as np
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print("\n[2/6] Importing CARLA...")
    import carla
    print("   CARLA imported successfully")
    
    print("\n[3/6] Importing source modules...")
    from source.sac_trainer import Trainer
    from source.sac import config, device
    from source.carlaenv import CarlaEnv
    from source.utility import setup_seed
    print(f"   Config loaded: {list(config.keys())}")
    
    print("\n[4/6] Setting up seed...")
    setup_seed(20)
    print("   Seed set to 20")
    
    print("\n[5/6] Creating CARLA environment...")
    env = CarlaEnv()
    print(f"   Environment created successfully")
    print(f"   World: {env.world.get_map().name}")
    
    print("\n[6/6] Reloading world...")
    env.client.reload_world(False)
    print("   World reloaded")
    
    print("\n[7/6] Creating trainer...")
    trainer = Trainer(env)
    print("   Trainer created successfully")
    
    print("\n" + "=" * 50)
    print("All initialization steps completed!")
    print("Starting training loop...")
    print("=" * 50 + "\n")
    
    from tqdm import tqdm
    for epoch_i in tqdm(range(config['epoch']), desc="Epoch"):
        trainer.train(epoch_i)
    
    print("\nTraining completed!")
    trainer.env.exit_env()
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print("\n" + "=" * 50)
    print("ERROR OCCURRED!")
    print("=" * 50)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
