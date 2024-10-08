import tensorflow as tf
import albumentations as A
import numpy as np
from tf_dataset_creation import train_dataset, val_dataset
import matplotlib.pyplot as plt

MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
#these values of RGB are provided in the paper

def preprocess(im_path, anno_path):
    img = tf.io.decode_jpeg(tf.io.read_file(im_path))
    img = tf.cast(img, tf.float32)
    img = (img-MEAN)/STD

    anno = tf.io.decode_jpeg(tf.io.read_file(anno_path))
    anno = tf.cast(tf.squeeze(anno, -1), tf.float32)

    return img, anno

prep_train_ds = (
    train_dataset
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
)

prep_val_ds = (
    val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
)

for i,j in prep_train_ds.take(1):
    print(i.shape, j.shape)


H,W = 512, 512


transform = A.Compose([
    A.RandomCrop(H,W, p=1.0),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.Transpose(p=0.3),
    A.Sharpen(alpha = (0.2, 0.5), lightness=(0.5, 1.0), p=0.1),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                   num_shadows_lower=1, num_shadows_upper=2,
                   shadow_dimension=5, p=0.1),
    A.RandomBrightnessContrast(p=0.2)
])

val_transform = A.Compose([
    A.Resize(H,W)
])


def aug_albument(image, mask):
    augmented = transform(image=image, mask=mask)
    return [tf.convert_to_tensor(augmented["image"], dtype=tf.float32),
            tf.convert_to_tensor(augmented["mask"], dtype=tf.float32)]

def val_aug_albument(image, mask):
    augmented = val_transform(image=image, mask=mask)
    return[tf.convert_to_tensor(augmented["image"], dtype=tf.float32),
           tf.convert_to_tensor(augmented["mask"], dtype=tf.float32)]


def augment(image, mask):
    aug_output = tf.numpy_function(func=aug_albument, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return{"pixel_values":tf.transpose(aug_output[0], (2,0,1)), "labels":aug_output[1]}

def val_augment(image, mask):
    aug_output = tf.numpy_function(func=val_aug_albument, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return{"pixel_values":tf.transpose(aug_output[0], (2,0,1)), "labels":aug_output[1]}

BATCH_SIZE = 1   ##changing baith_size to 1
train_ds = (
    prep_train_ds
    .shuffle(10)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    prep_val_ds
    .map(val_augment)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# for i in train_ds.take(1):
#     print(i)