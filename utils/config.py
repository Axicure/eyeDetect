import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_IMG_DIR = os.path.join(DATA_DIR, 'processed_images')

# 模型参数
MODEL_CONFIG = {
    'input_shape': (299, 299, 3),
    'pretrained_weights': 'imagenet',
    'trainable_layers': 250,
    'num_classes': 8,
    'class_names': ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # 确保顺序正确
}

# 训练参数
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'validation_split': 0.2  # 删除 class_weights
}

# 推理参数（无需修改）
INFERENCE_CONFIG = {
    'confidence_threshold': 0.65,
    'intermediate_output_dir': os.path.join(BASE_DIR, 'inference_pipeline/intermediate_outputs')
}