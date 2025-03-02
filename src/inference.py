import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.config import *
from tensorflow.keras.models import load_model


class InferencePipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )

    def _save_intermediate(self, img, stage, filename):
        output_dir = os.path.join(INFERENCE_CONFIG['intermediate_output_dir'], stage)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def process_single_eye(self, img_path, save_intermediate=False):
        # 预处理
        raw_img = cv2.imread(img_path)
        preprocessed = cv2.resize(raw_img, (299, 299))
        if save_intermediate:
            self._save_intermediate(preprocessed, '0_preprocessed', os.path.basename(img_path))

        # 特征提取
        normalized = preprocessed / 255.0
        features = self.feature_extractor.predict(np.expand_dims(normalized, axis=0))

        # 可视化特征图
        if save_intermediate:
            self._plot_activation_maps(features[0], os.path.basename(img_path))

        # 最终预测
        prediction = self.model.predict(np.expand_dims(normalized, axis=0))
        return (prediction > INFERENCE_CONFIG['confidence_threshold']).astype(int)[0]

    def _plot_activation_maps(self, features, filename):
        plt.figure(figsize=(20, 20))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(features[:, :, i], cmap='viridis')
            plt.axis('off')
        output_path = os.path.join(
            INFERENCE_CONFIG['intermediate_output_dir'],
            '2_activation_maps',
            f"{filename.split('.')[0]}_features.png"
        )
        plt.savefig(output_path)
        plt.close()


def main(img_dir, save_intermediate=False):
    pipeline = InferencePipeline('best_model.h5')
    results = []

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file))
        left_pred = pipeline.process_single_eye(img_path, save_intermediate)
        # 右眼处理逻辑相同

        # 合并双眼结果
        final_labels = left_pred | right_pred  # 按位或操作
        results.append({
            'filename': img_file,
            'prediction': ''.join(str(int(b)) for b in final_labels)
        })

        pd.DataFrame(results).to_csv('inference_pipeline/final_predictions.csv', index=False)

        if __name__ == '__main__':
            import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_dir', default='inference_pipeline/input_images')
        parser.add_argument('--save_intermediate', action='store_true')
        args = parser.parse_args()

        main(args.image_dir, args.save_intermediate)