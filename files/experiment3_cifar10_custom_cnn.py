# -*- coding: utf-8 -*-
"""
Experiment 3: CIFAR-10 classification with a small CNN trained from scratch,
plus a Keras ImageDataGenerator augmentation demo.

Extracted and cleaned up from the original notebook export (source lines ~685-753).
This was the most self-contained section of the original file -- no undefined
variables -- but still shared/overwrote the `model` variable used by the other
experiments, which is resolved simply by this now being its own script.

Bugs fixed vs. the original:
  - `datagen.flow(...)` is an infinite generator; the original relied on a
    manual `if i > 4: break` with no other safety net. Kept the same limit
    but made it an explicit named constant (`NUM_AUGMENTED_SAMPLES`) instead
    of a magic number, so the intent is clear.
  - Removed the duplicate `model.summary()` calls and stray commented-out
    lines from the original cell.

Requirements (install before running):
    pip install tensorflow matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
EPOCHS = 4
NUM_AUGMENTED_SAMPLES = 5


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Labels come back as shape (N, 1); flatten for sparse_categorical losses / indexing.
    train_labels, test_labels = train_labels.flatten(), test_labels.flatten()
    return train_images, train_labels, test_images, test_labels


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))  # logits, 10 classes

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def demo_augmentation(train_images, image_index=20):
    """Show a handful of augmented variants of one training image."""
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    img = img_to_array(train_images[image_index])
    img = img.reshape((1,) + img.shape)  # add batch dim

    for i, batch in enumerate(datagen.flow(img)):
        plt.figure(i)
        plt.imshow(img_to_array(batch[0]))
        if i + 1 >= NUM_AUGMENTED_SAMPLES:
            break
    plt.show()


def main():
    train_images, train_labels, test_images, test_labels = load_data()

    img_index = 7
    plt.imshow(train_images[img_index], cmap=plt.cm.binary)
    plt.xlabel(CLASS_NAMES[train_labels[img_index]])
    plt.show()

    model = build_model()
    model.summary()

    model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    demo_augmentation(train_images)


if __name__ == "__main__":
    main()
