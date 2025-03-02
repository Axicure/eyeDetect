import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.config import *


def build_model():
    base_model = InceptionV3(
        weights=MODEL_CONFIG['pretrained_weights'],
        include_top=False,
        input_shape=MODEL_CONFIG['input_shape']
    )

    # 冻结前200层
    for layer in base_model.layers[:MODEL_CONFIG['trainable_layers']]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(MODEL_CONFIG['num_classes'], activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def train():
    # 加载预处理数据
    metadata = pd.read_csv(os.path.join(DATA_DIR, 'processed_metadata.csv'))

    # 构建数据管道
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=TRAIN_CONFIG['validation_split']
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=metadata,
        directory=PROCESSED_IMG_DIR,
        x_col='left_img',
        y_col='labels',
        class_mode='multi_output',
        batch_size=TRAIN_CONFIG['batch_size']
    )

    # 编译模型
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(TRAIN_CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # 训练
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )

    model.fit(
        train_generator,
        epochs=TRAIN_CONFIG['epochs'],
        callbacks=[checkpoint],
        class_weight=TRAIN_CONFIG['class_weights']
    )


if __name__ == '__main__':
    train()