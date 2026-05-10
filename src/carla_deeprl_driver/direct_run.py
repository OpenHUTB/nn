import sys
import os

os.chdir(r'c:\Users\12698\Desktop\f\carla_RL')
sys.path.insert(0, r'c:\Users\12698\Desktop\f\carla_RL')

print("Current directory:", os.getcwd())
print("Python version:", sys.version)
print("\n" + "=" * 50)
print("Starting CARLA SAC Training")
print("=" * 50 + "\n")

try:
    exec(open('run_sac.py').read(), {'__name__': '__main__'})
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
except Exception as e:
    print(f"\n\nError occurred: {e}")
    import traceback
    traceback.print_exc()
