import tensorflow as tf
import os

im_path = "../dataset/png_images/IMAGES/"
anno_path = "../dataset/png_masks/MASKS/"
val_im_path = "../dataset/val_dataset/png_images/IMAGES/"
val_anno_path = "../dataset/val_dataset/png_masks/MASKS/"

train_dataset = tf.data.Dataset.from_tensor_slices(
    ([im_path+i for i in os.listdir(im_path)],
     [anno_path + "seg" + i[3:] for i in os.listdir(im_path)])
)
for i in os.listdir(im_path):
    print(i)
    break

val_dataset = tf.data.Dataset.from_tensor_slices(
    ([val_im_path+i for i in os.listdir(val_im_path)],
     [val_anno_path+"seg"+i[3:] for i in os.listdir(val_im_path)])
)

for i in train_dataset.take(3):
    print(i)

for i in val_dataset.take(3):
    print(i)