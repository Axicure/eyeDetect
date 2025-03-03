import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from utils.config import *


class LabelGenerator:
    """根据keywords动态生成多标签"""

    def __init__(self):
        # 加载标签映射规则
        with open(os.path.join(DATA_DIR, '..', 'utils', 'label_mapping.json')) as f:
            self.mapping = json.load(f)

        # 建立类别索引
        self.class_indices = {name: idx for idx, name in enumerate(MODEL_CONFIG['class_names'])}

    def keywords_to_labels(self, keywords_str):
        """将关键词字符串转换为二进制标签数组"""
        labels = np.zeros(len(MODEL_CONFIG['class_names']), dtype=np.float32)

        # 分割关键词
        keywords = [kw.strip().lower() for kw in keywords_str.split('，')]

        # 匹配映射规则
        for kw in keywords:
            matched = False
            for pattern, codes in self.mapping.items():
                if pattern in kw:
                    for code in codes:
                        labels[self.class_indices[code]] = 1.0
                    matched = True
            if not matched and '_default' in self.mapping:
                for code in self.mapping['_default']:
                    labels[self.class_indices[code]] = 1.0
        return labels


def build_model():
    """构建InceptionV3多标签分类模型"""
    base_model = InceptionV3(
        weights=MODEL_CONFIG['pretrained_weights'],
        include_top=False,
        input_shape=MODEL_CONFIG['input_shape']
    )

    # 冻结基础模型层
    for layer in base_model.layers[:MODEL_CONFIG['trainable_layers']]:
        layer.trainable = False

    # 添加自定义层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(MODEL_CONFIG['num_classes'], activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=outputs)


class CustomDataGenerator(tf.keras.utils.Sequence):
    """自定义数据生成器"""

    def __init__(self, df, img_dir, label_generator, batch_size=32, shuffle=True):
        self.df = df
        self.img_dir = img_dir
        self.label_generator = label_generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images = []
        labels = []
        for _, row in batch_df.iterrows():
            # 加载图像
            img_path = os.path.join(self.img_dir, row['processed_image'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=MODEL_CONFIG['input_shape'][:2])
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

            # 生成标签
            label = self.label_generator.keywords_to_labels(row['keywords'])

            images.append(img_array)
            labels.append(label)

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train():
    # 加载数据
    metadata = pd.read_csv(os.path.join(DATA_DIR, 'processed_metadata.csv'))

    # 初始化标签生成器
    label_gen = LabelGenerator()

    # 划分训练验证集
    train_df = metadata.sample(frac=1 - TRAIN_CONFIG['validation_split'], random_state=42)
    val_df = metadata.drop(train_df.index)

    # 创建数据生成器
    train_gen = CustomDataGenerator(
        train_df,
        PROCESSED_IMG_DIR,
        label_gen,
        batch_size=TRAIN_CONFIG['batch_size']
    )

    val_gen = CustomDataGenerator(
        val_df,
        PROCESSED_IMG_DIR,
        label_gen,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False
    )

    # 计算类别权重
    all_labels = np.array([label_gen.keywords_to_labels(kw) for kw in metadata['keywords']])
    class_weights = {}
    for idx in range(all_labels.shape[1]):
        positive = np.sum(all_labels[:, idx])
        total = len(all_labels)
        weight = (total - positive) / positive if positive > 0 else 1.0
        class_weights[idx] = weight

    # 构建模型
    model = build_model()

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAIN_CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 回调函数
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True
        )
    ]

    # 开始训练
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TRAIN_CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=class_weights
    )


if __name__ == '__main__':
    train()