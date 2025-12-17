import argparse
import os
from gesture_classifier import GestureClassifier, EnsembleGestureClassifier
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='训练手势识别模型')
    parser.add_argument('--dataset', type=str, default='dataset/processed/gesture_dataset.pkl',
                        help='数据集路径')
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['svm', 'random_forest', 'mlp', 'ensemble'],
                        help='模型类型')
    parser.add_argument('--output_dir', type=str, default='dataset/models',
                        help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--collect_data', action='store_true',
                        help='是否先收集数据')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 如果需要收集数据
    if args.collect_data:
        from gesture_data_collector import GestureDataCollector
        print("启动数据收集...")
        collector = GestureDataCollector()
        collector.run_collection_app()

    if args.model_type == 'ensemble':
        # 训练多个模型并集成
        models = []
        model_types = ['svm', 'random_forest', 'mlp']

        for model_type in model_types:
            print(f"\n训练 {model_type} 模型...")
            classifier = GestureClassifier(model_type=model_type)

            model_path = os.path.join(args.output_dir, f'gesture_{model_type}.pkl')
            accuracy = classifier.train(args.dataset, args.test_size, model_path)

            models.append(model_path)
            print(f"{model_type} 准确率: {accuracy:.4f}")

        # 创建集成模型
        print("\n创建集成模型...")
        ensemble = EnsembleGestureClassifier(models)

        # 保存集成模型信息
        ensemble_info = {
            'model_paths': models,
            'model_types': model_types
        }

        import joblib
        ensemble_path = os.path.join(args.output_dir, 'gesture_ensemble.pkl')
        joblib.dump(ensemble_info, ensemble_path)
        print(f"集成模型信息已保存到: {ensemble_path}")

    else:
        # 训练单个模型
        print(f"训练 {args.model_type} 模型...")
        classifier = GestureClassifier(model_type=args.model_type)

        model_path = os.path.join(args.output_dir, f'gesture_{args.model_type}.pkl')
        accuracy = classifier.train(args.dataset, args.test_size, model_path)

        print(f"\n训练完成!")
        print(f"模型类型: {args.model_type}")
        print(f"准确率: {accuracy:.4f}")
        print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()