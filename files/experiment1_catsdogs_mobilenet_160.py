# -*- coding: utf-8 -*-
"""
Experiment 1: Cats vs Dogs classification via MobileNetV2 transfer learning (160x160)

Extracted and cleaned up from the original notebook export (source lines ~1-216).

Bugs fixed vs. the original:
  - Classification head was `Dense(10, activation='softmax')` even though
    cats_vs_dogs is a 2-class problem (the "10 classes" comment was left over
    from a CIFAR-10 experiment pasted into the same cell). Fixed to Dense(2).
  - `raw_validation` was `.repeat(3)`'d before `model.fit(...)`, which turns
    validation into an effectively unbounded dataset with no matching
    `validation_steps` -- removed the repeat.
  - Removed the Colab-only `!pip install` shell magic; see requirements note
    below instead.
  - Removed ~10 duplicate "grab one image and plot it" blocks; kept one.

Requirements (install before running):
    pip install tensorflow tensorflow-datasets tf-models-official matplotlib numpy
"""

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 160
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 10000
EPOCHS = 2


def load_data():
    """80/20 split manually done as 60/20/20 train/val/test."""
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:60%]", "train[60%:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
    )
    get_label_name = metadata.features["label"].int2str
    return raw_train, raw_validation, raw_test, get_label_name


def preprocess_image(image, label):
    """Resize (with padding to preserve aspect ratio) and normalize for MobileNetV2."""
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


rotation_layer = layers.RandomRotation(factor=0.2)


def augment_data(image, label):
    """Random rotation augmentation, applied to training data only."""
    image = rotation_layer(image)
    return image, label


def build_datasets(raw_train, raw_validation, raw_test):
    train_ds = (
        raw_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        raw_validation.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        raw_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for feature extraction

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)  # 2 classes: cat, dog

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def demo_rotation_augmentation(image, label, get_label_name):
    """Standalone demo of tfm.vision.augment.rotate on a single image, then
    resize/cast it back to a form the model can predict on."""
    rotated_image = tfm.vision.augment.rotate(image, 180.0)

    res_image = tf.image.resize_with_pad(rotated_image, IMG_SIZE, IMG_SIZE)
    resized_image_uint8 = tf.cast(res_image, tf.uint8)

    plt.figure()
    plt.imshow(resized_image_uint8.numpy())
    plt.xlabel(get_label_name(label))
    plt.title("Rotated 180 degrees, resized to model input size")
    plt.savefig("rotated.png")
    plt.show(block=False)
    plt.pause(1)

    return res_image


def predict_single_image(model, image, label, get_label_name):
    """image is expected already resized/preprocessed to (IMG_SIZE, IMG_SIZE, 3)."""
    batch = np.expand_dims(image, axis=0)
    predicted_label = get_label_name(model.predict(batch).argmax())
    original_label = get_label_name(label)
    print(f"Original label is {original_label} and predicted label is {predicted_label}")


def main():
    raw_train, raw_validation, raw_test, get_label_name = load_data()

    # Peek at two raw images before preprocessing
    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
        plt.show()

    train_ds, val_ds, test_ds = build_datasets(raw_train, raw_validation, raw_test)

    model = build_model()
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    model.evaluate(test_ds, verbose=2)

    # Rotation-augmentation demo + prediction, on one raw test image
    for image, label in raw_test.take(1):
        pass
    rotated_and_resized = demo_rotation_augmentation(image, label, get_label_name)
    predict_single_image(model, rotated_and_resized.numpy(), label, get_label_name)


if __name__ == "__main__":
    main()
