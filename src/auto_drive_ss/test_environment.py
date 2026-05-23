import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    print("Testing imports...")
    
    # Test core dependencies
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import carla
        print("✓ CARLA: imported successfully")
    except ImportError as e:
        print(f"✗ CARLA import failed: {e}")
        return False
    
    try:
        import gym
        print(f"✓ Gym: {gym.__version__}")
    except ImportError as e:
        print(f"✗ Gym import failed: {e}")
        return False
    
    try:
        import timm
        print(f"✓ TIMM: {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM import failed: {e}")
        return False
    
    try:
        import einops
        print(f"✓ Einops: {einops.__version__}")
    except ImportError as e:
        print(f"✗ Einops import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    # Test local modules
    try:
        from src.models.perception_system import PerceptionSystem
        print("✓ PerceptionSystem imported successfully")
    except ImportError as e:
        print(f"✗ PerceptionSystem import failed: {e}")
        return False
    
    try:
        from src.models.rgb_encoder import RGBEncoder
        print("✓ RGBEncoder imported successfully")
    except ImportError as e:
        print(f"✗ RGBEncoder import failed: {e}")
        return False
    
    try:
        from src.models.depth_encoder import DepthEncoder
        print("✓ DepthEncoder imported successfully")
    except ImportError as e:
        print(f"✗ DepthEncoder import failed: {e}")
        return False
    
    # Test CUDA availability
    try:
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, using CPU")
    except Exception as e:
        print(f"⚠ CUDA check failed: {e}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Environment Test for Autonomous Driving SSL-RL Project")
    print("=" * 60)
    
    success = test_imports()
    
    print("=" * 60)
    if success:
        print("✅ All imports successful!")
        print("\nTo run the project:")
        print("1. Start CARLA simulator first:")
        print("   - Download CARLA from https://carla.org/")
        print("   - Run: ./CarlaUE4.exe (Windows) or ./CarlaUE4.sh (Linux)")
        print("   - CARLA will listen on port 2000")
        print("\n2. Run SSL pre-training (requires dataset):")
        print("   python src/ssl_trainer.py")
        print("\n3. Run RL training with SSL:")
        print("   python src/rl_training_with_ssl.py")
        print("\n4. Evaluate trained model:")
        print("   python src/evaluate_rl.py --checkpoint <path_to_checkpoint>")
    else:
        print("❌ Some imports failed. Please check your environment.")
    print("=" * 60)