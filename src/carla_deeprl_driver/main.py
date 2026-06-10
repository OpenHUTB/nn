from source.trainer import Trainer
from source.utility import setup_seed

if __name__ == '__main__':
    setup_seed(20)
    trainer = Trainer()
    trainer.env.client.reload_world(False)
    trainer.training_loop()
    trainer.env.exit_env()
