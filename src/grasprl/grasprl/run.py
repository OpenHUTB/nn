import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from trainer.dqn_baseline import main

if __name__ == "__main__":
    main()
