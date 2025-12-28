import subprocess
import sys

def check_and_install():
    """检查并安装缺失的依赖"""
    required_packages = [
        "torch",
        "torchvision",
        "opencv-python",
        "Pillow",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "plotly"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n📦 需要安装 {len(missing_packages)} 个包: {', '.join(missing_packages)}")
        choice = input("是否现在安装？(y/n): ")
        
        if choice.lower() == 'y':
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ {package} 安装成功")
                except subprocess.CalledProcessError:
                    print(f"❌ {package} 安装失败")
        else:
            print("请手动安装缺失的依赖包")
    else:
        print("\n🎉 所有依赖包都已安装！")

if __name__ == "__main__":
    check_and_install()