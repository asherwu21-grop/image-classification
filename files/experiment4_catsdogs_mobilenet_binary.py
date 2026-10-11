# -*- coding: utf-8 -*-
"""
Experiment 4: Cats vs Dogs binary classification via MobileNetV2 transfer
learning, using a single-logit Dense head + BinaryCrossentropy.

Extracted and cleaned up from the original notebook export (source lines ~754-872).
This was the closest to being a correct, self-contained pipeline in the
original file -- the main problems were structural (variable names reused
from experiments 1-3) and an IndentationError a few hundred lines earlier
in the original that would have prevented Python from ever reaching this
section at all when run as a plain script.

Bugs fixed vs. the original:
  - Removed duplicate "peek at N images" blocks (there were two nearly
    identical loops back to back).
  - Made batching order deterministic: preprocess -> shuffle -> batch, so
    shuffling happens on individual examples rather than in whatever order
    `.map()`/`.batch()` were previously interleaved.

Requirements (install before running):
    pip install tensorflow tensorflow-datasets matplotlib numpy
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 160
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
INITIAL_EPOCHS = 3
VALIDATION_STEPS = 20
BASE_LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "dogs_vs_cats.h5"


def load_data():
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
        as_supervised=True,
    )
    get_label_name = metadata.features["label"].int2str
    return raw_train, raw_validation, raw_test, get_label_name


def format_example(image, label):
    """Scale pixels to [-1, 1] and resize to IMG_SIZE."""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def build_datasets(raw_train, raw_validation, raw_test):
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)
    return train_batches, validation_batches, test_batches


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)  # single logit for binary classification

    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=BASE_LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    raw_train, raw_validation, raw_test, get_label_name = load_data()

    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
        plt.show()

    train_batches, validation_batches, test_batches = build_datasets(
        raw_train, raw_validation, raw_test
    )

    model = build_model()
    model.summary()

    # Baseline evaluation before any fine-tuning
    loss0, accuracy0 = model.evaluate(validation_batches, steps=VALIDATION_STEPS)
    print(f"Pre-training loss: {loss0:.4f}, accuracy: {accuracy0:.4f}")

    history = model.fit(
        train_batches,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_batches,
    )
    print("Training accuracy by epoch:", history.history["accuracy"])

    test_loss, test_acc = model.evaluate(test_batches)
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    model.save(MODEL_SAVE_PATH)
    reloaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"Model saved to and reloaded from {MODEL_SAVE_PATH}")
    return reloaded_model


if __name__ == "__main__":
    main()
