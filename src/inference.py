import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from utils.config import INFERENCE_CONFIG, MODEL_CONFIG


class InferencePipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )

    def _save_intermediate(self, img, stage, filename):
        """保存中间结果到指定目录"""
        output_dir = os.path.join(INFERENCE_CONFIG['intermediate_output_dir'], stage)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _plot_activation_maps(self, features, filename):
        """可视化特征图"""
        plt.figure(figsize=(20, 20))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(features[0, :, :, i], cmap='viridis')
            plt.axis('off')

        output_path = os.path.join(
            INFERENCE_CONFIG['intermediate_output_dir'],
            '2_activation_maps',
            f"{os.path.splitext(filename)[0]}_features.png"
        )
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def process_single_eye(self, img_path, save_intermediate=False):
        """处理单张眼底图片"""
        # 读取原始图片
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")

        # 预处理并保存中间结果
        preprocessed = cv2.resize(raw_img, MODEL_CONFIG['input_shape'][:2])
        if save_intermediate:
            self._save_intermediate(preprocessed, '0_preprocessed', os.path.basename(img_path))

        # 归一化处理
        normalized = preprocessed.astype(np.float32) / 255.0

        # 特征提取
        features = self.feature_extractor.predict(np.expand_dims(normalized, axis=0))

        # 可视化特征图
        if save_intermediate:
            self._plot_activation_maps(features, os.path.basename(img_path))

        # 最终预测
        prediction = self.model.predict(np.expand_dims(normalized, axis=0))
        return (prediction > INFERENCE_CONFIG['confidence_threshold']).astype(int)[0]


def merge_predictions(left_pred, right_pred):
    """合并双眼预测结果"""
    final = np.zeros_like(left_pred)
    # 糖尿病(D): 任意眼阳性即标记
    final[1] = left_pred[1] or right_pred[1]
    # 白内障(C): 同上
    final[3] = left_pred[3] or right_pred[3]
    # 其他疾病按相同逻辑处理...
    return final


def main(image_dir, save_intermediate=False):
    pipeline = InferencePipeline('best_model.h5')
    results = []

    for img_file in os.listdir(image_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_file)

        # 假设左眼图片文件名包含'_left'，右眼包含'_right'
        if '_left' in img_file:
            left_pred = pipeline.process_single_eye(img_path, save_intermediate)
        elif '_right' in img_file:
            right_pred = pipeline.process_single_eye(img_path, save_intermediate)
        else:
            continue  # 忽略不符合命名规范的文件

    # 合并双眼结果（需根据实际数据组织方式调整）
    final_labels = merge_predictions(left_pred, right_pred)
    results.append({
        'ID': os.path.splitext(img_file)[0],
        'N': final_labels[0],
        'D': final_labels[1],
        'G': final_labels[2],
        'C': final_labels[3],
        'A': final_labels[4],
        'H': final_labels[5],
        'M': final_labels[6],
        'O': final_labels[7]
    })

    # 保存最终结果
    pd.DataFrame(results).to_csv(
        os.path.join(INFERENCE_CONFIG['intermediate_output_dir'], '../final_predictions.csv'),
        index=False
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='眼底影像疾病分类推理')
    parser.add_argument('--image_dir',
                        default='inference_pipeline/input_images',
                        help='待预测图片目录路径')
    parser.add_argument('--save_intermediate',
                        action='store_true',
                        help='是否保存中间处理结果')
    args = parser.parse_args()

    main(args.image_dir, args.save_intermediate)