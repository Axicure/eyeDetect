import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
from utils.config import DATA_DIR, PROCESSED_IMG_DIR, MODEL_CONFIG


def load_label_mapping():
    with open(os.path.join(DATA_DIR, '..', 'utils', 'label_mapping.json')) as f:
        return json.load(f)


def preprocess_image(img_path, save=False):
    """图像预处理与保存"""
    img = cv2.imread(os.path.join(DATA_DIR, 'raw_images', img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    if save:
        output_path = os.path.join(PROCESSED_IMG_DIR, img_path)
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img / 255.0


def generate_labels(keywords, mapping):
    """将关键词转换为二进制标签"""
    labels = [0] * 8
    for kw in keywords.split('，'):
        found = False
        for pattern, codes in mapping.items():
            if pattern in kw.lower():
                for code in codes:
                    idx = MODEL_CONFIG['class_names'].index(code)
                    labels[idx] = 1
                found = True
        if not found and '_default' in mapping:
            for code in mapping['_default']:
                idx = MODEL_CONFIG['class_names'].index(code)
                labels[idx] = 1
    return labels


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    mapping = load_label_mapping()

    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 处理左眼
        if not pd.isna(row['Left-Fundus']):
            preprocess_image(row['Left-Fundus'], save=True)
            left_labels = generate_labels(row['Left-Diagnostic Keywords'], mapping)

        # 处理右眼同理...

        processed_data.append({
            'left_img': row['Left-Fundus'],
            'right_img': row['Right-Fundus'],
            'labels': left_labels + right_labels  # 合并双眼标签
        })

    pd.DataFrame(processed_data).to_csv(os.path.join(DATA_DIR, 'processed_metadata.csv'), index=False)


if __name__ == '__main__':
    main()