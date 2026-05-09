import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from gesture_classifier import GestureClassifier
from deep_gesture_classifier import DeepGestureClassifier

def plot_training_history(results, save_path='training_comparison.png'):
    """绘制训练历史对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].set_title('训练损失曲线')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].set_title('测试准确率曲线')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for i, (model_name, data) in enumerate(results.items()):
        if 'losses' in data:
            axes[0].plot(data['losses'], label=model_name, color=colors[i % len(colors)])
        if 'test_accs' in data:
            axes[1].plot(data['test_accs'], label=model_name, color=colors[i % len(colors)], marker='o', markersize=2)
        elif 'accuracy' in data:
            axes[1].axhline(y=data['accuracy'], label=model_name, color=colors[i % len(colors)], linestyle='--')
    
    axes[0].legend()
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='训练深度学习手势识别模型')
    parser.add_argument('--dataset', type=str, default='dataset/processed/gesture_dataset.pkl',
                        help='数据集路径')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'transformer', 'mlp', 'all'],
                        help='模型类型')
    parser.add_argument('--output_dir', type=str, default='dataset/models',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--compare', action='store_true',
                        help='是否对比所有模型')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    if args.compare or args.model_type == 'all':
        print("=== 训练所有模型进行对比 ===")
        
        # 传统机器学习模型
        print("\n--- 训练传统ML模型 ---")
        ml_models = ['svm', 'random_forest', 'mlp']
        for model_type in ml_models:
            print(f"\n训练 {model_type}...")
            classifier = GestureClassifier(model_type=model_type)
            model_path = os.path.join(args.output_dir, f'gesture_{model_type}.pkl')
            accuracy = classifier.train(args.dataset, args.test_size, model_path)
            results[model_type] = {'accuracy': accuracy}
        
        # 深度学习模型
        print("\n--- 训练深度学习模型 ---")
        deep_models = ['cnn', 'transformer', 'mlp']
        for model_type in deep_models:
            print(f"\n训练 Deep {model_type}...")
            classifier = DeepGestureClassifier(model_type=model_type)
            model_path = os.path.join(args.output_dir, f'gesture_deep_{model_type}.pth')
            accuracy, losses, train_accs, test_accs = classifier.train(
                args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, test_size=args.test_size, save_path=model_path
            )
            results[f'Deep {model_type}'] = {
                'accuracy': accuracy,
                'losses': losses,
                'train_accs': train_accs,
                'test_accs': test_accs
            }
        
        # 绘制对比图
        print("\n=== 绘制训练对比图 ===")
        plot_path = os.path.join(args.output_dir, 'training_comparison.png')
        plot_training_history(results, plot_path)
        
        # 打印对比结果
        print("\n=== 模型对比结果 ===")
        print(f"{'模型':<20} {'准确率':<10}")
        print("-" * 30)
        for model_name, data in results.items():
            print(f"{model_name:<20} {data['accuracy']:<10.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n最优模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")
    
    else:
        print(f"训练 Deep {args.model_type} 模型...")
        classifier = DeepGestureClassifier(model_type=args.model_type)
        model_path = os.path.join(args.output_dir, f'gesture_deep_{args.model_type}.pth')
        
        accuracy, losses, train_accs, test_accs = classifier.train(
            args.dataset, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, test_size=args.test_size, save_path=model_path
        )
        
        # 绘制训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(losses, label='训练损失')
        axes[0].set_title('训练损失曲线')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(train_accs, label='训练准确率')
        axes[1].plot(test_accs, label='测试准确率')
        axes[1].set_title('准确率曲线')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f'training_{args.model_type}.png')
        plt.savefig(plot_path, dpi=150)
        plt.show()
        
        print(f"\n训练完成!")
        print(f"模型类型: Deep {args.model_type}")
        print(f"准确率: {accuracy:.4f}")
        print(f"模型已保存到: {model_path}")
        print(f"训练曲线已保存到: {plot_path}")

if __name__ == "__main__":
    main()