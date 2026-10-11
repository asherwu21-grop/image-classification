# -*- coding: utf-8 -*-
"""
Experiment 2: CIFAR-10 classification via MobileNetV2 transfer learning (96x96)

Extracted and cleaned up from the original notebook export (source lines ~218-453).

Bugs fixed vs. the original:
  - `ds_train`, `ds_test`, and `ds_info` were referenced but NEVER assigned
    anywhere in the original file (the `tfds.load('cifar10', ...)` call that
    would have created them was left sitting inside a triple-quoted, non-
    executing docstring). Added a real `tfds.load` call here.
  - `class_n` was used but never defined in this section either -- added it.
  - Added `.cache()` after preprocessing, which the original omitted for this
    pipeline (present only in a commented-out block), so every epoch was
    re-decoding/re-resizing images from scratch.
  - Removed ~6 duplicate "grab an image, predict, compare labels" blocks that
    used inconsistent reshape targets (32x32 vs 96x96) left over from being
    copy-pasted across experiments; kept one correct version.

Requirements (install before running):
    pip install tensorflow tensorflow-datasets matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 96
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10000
EPOCHS = 3

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_data():
    (ds_train, ds_test), ds_info = tfds.load(
        "cifar10",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test, ds_info


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def build_datasets(ds_train, ds_test):
    train_ds = (
        ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)  # 10 CIFAR-10 classes

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_single_image(model, image, label):
    """image: raw HWC array/tensor at any size; will be resized to IMG_SIZE."""
    resized = tf.image.resize(np.array(image), [IMG_SIZE, IMG_SIZE])
    batch = np.expand_dims(resized, axis=0)
    predicted_label = CLASS_NAMES[model.predict(batch).argmax()]
    original_label = CLASS_NAMES[int(label)]
    print(f"Original label is {original_label} and predicted label is {predicted_label}")


def main():
    ds_train, ds_test, ds_info = load_data()

    for image, label in ds_test.take(1):
        first_image, first_label = image, label
    plt.figure()
    plt.imshow(first_image)
    plt.xlabel(CLASS_NAMES[int(first_label)])
    plt.show()

    train_ds, test_ds = build_datasets(ds_train, ds_test)

    model = build_model()
    model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)
    model.evaluate(test_ds, verbose=2, return_dict=True)

    predict_single_image(model, first_image.numpy(), first_label.numpy())


if __name__ == "__main__":
    main()
