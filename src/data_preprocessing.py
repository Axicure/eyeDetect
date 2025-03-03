import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
from utils.config import DATA_DIR, PROCESSED_IMG_DIR, MODEL_CONFIG


def load_label_mapping():
    """加载标签映射规则"""
    with open(os.path.join(DATA_DIR, '..', 'utils', 'label_mapping.json')) as f:
        return json.load(f)


def preprocess_and_save(img_path):
    """图像预处理并保存新文件"""
    # 读取原始图片
    raw_img = cv2.imread(os.path.join(DATA_DIR, 'raw_images', img_path))
    if raw_img is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")

    # 预处理流程
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    # 生成新文件名
    base_name = os.path.splitext(img_path)[0]
    new_filename = f"{base_name}_processed.jpg"
    output_path = os.path.join(PROCESSED_IMG_DIR, new_filename)

    # 保存处理后的图片
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return new_filename


def generate_single_eye_labels(keywords, mapping):
    """为单眼生成标签"""
    labels = [0] * 8
    for kw in keywords.split('，'):
        matched = False
        for pattern, codes in mapping.items():
            if pattern in kw.lower():
                for code in codes:
                    idx = MODEL_CONFIG['class_names'].index(code)
                    labels[idx] = 1
                matched = True
        if not matched and '_default' in mapping:
            for code in mapping['_default']:
                idx = MODEL_CONFIG['class_names'].index(code)
                labels[idx] = 1
    return labels


def process_row(row, mapping):
    """处理单行数据生成两个眼睛记录"""
    records = []

    # 处理左眼
    if pd.notna(row['Left-Fundus']):
        processed_img = preprocess_and_save(row['Left-Fundus'])
        left_labels = generate_single_eye_labels(row['Left-Diagnostic Keywords'], mapping)
        records.append({
            'processed_image': processed_img,
            'keywords': row['Left-Diagnostic Keywords']
        })

    # 处理右眼
    if pd.notna(row['Right-Fundus']):
        processed_img = preprocess_and_save(row['Right-Fundus'])
        right_labels = generate_single_eye_labels(row['Right-Diagnostic Keywords'], mapping)
        records.append({
            'processed_image': processed_img,
            'keywords': row['Right-Diagnostic Keywords']
        })

    return records


def main():
    # 创建输出目录
    os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

    # 加载数据
    df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    mapping = load_label_mapping()

    # 处理所有数据
    processed_records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed_records.extend(process_row(row, mapping))

    # 保存新的元数据
    new_df = pd.DataFrame(processed_records)
    new_df.to_csv(os.path.join(DATA_DIR, 'processed_metadata.csv'), index=False)
    print(f"生成 {len(new_df)} 条记录，保存至 {PROCESSED_IMG_DIR}")


if __name__ == '__main__':
    main()