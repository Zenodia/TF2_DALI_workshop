import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import tensorflow  as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.tf as dali_tf
tf.logging.set_verbosity(tf.logging.ERROR)

class Opt3Pipeline(Pipeline):
    def __init__(self, img_file_list,  num_shards, shard_id, batch_size, num_threads, device_id):
        super(Opt3Pipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input_img = ops.FileReader(file_root = "", num_shards=num_shards, shard_id=shard_id, file_list=img_file_list,random_shuffle = True, initial_fill = 21)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.rrc = ops.RandomResizedCrop(
                device="gpu",
                size=(800,800),
                random_area=[0.8, 0.8])
        self.flip_h = ops.Flip(device = "gpu", vertical = 0, horizontal = 1)
        self.rotate = ops.Rotate(device = "gpu", angle = 30, interp_type = types.INTERP_LINEAR, fill_value = 0)
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)    

    def define_graph(self):
        imgs, labels = self.input_img()
        images = self.decode(imgs)
        output = self.rrc(images)
        output = self.flip_h(output)
        output = self.rotate(output)
        output = self.resize(output)
        return (output, labels)

@profile
def get_pipe(device_id):
    pipe = Opt3Pipeline("bigimgs_path.txt", 2, 0, batch_size, 8, device_id)
    return pipe
@profile
def get_reset50_model():
    # load model without classifier layers
    num_class=3
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))

    # make all layers trainable
    for layer in base_model.layers:
        layer.trainable = True
    # add your head on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_class, activation='softmax')(x)

    m = Model(inputs=base_model.input, outputs=output)
    return m

@profile
def main(device_id):
    with tf.device('/gpu:{}'.format(str(device_id))):
        pipe=get_pipe(device_id)
        daliop = dali_tf.DALIIterator()
        # Define shapes and types of the outputs
        # Define shapes and types of the outputs
        shapes = [
            (batch_size, 224, 224,3),
            (batch_size,1)]
        dtypes = [
            tf.float32,
            tf.int32]

        # Create tensorflow dataset

        out_dataset = dali_tf.DALIDataset(
            pipeline=pipe,
            batch_size=batch_size,
            shapes=shapes,
            dtypes=dtypes,
            device_id=0)

        m=get_reset50_model()
        m.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # Train using DALI dataset
        m.fit(
            out_dataset,
            epochs=10,
            steps_per_epoch=8*8,
            use_multiprocessing=True )

if __name__ == "__main__":
    
    label2num={'notHappy':0,'Happy':1,'others':2}
    num2label=dict([(b,a) for (a,b) in label2num.items()])
    device_id=0
    batch_size=2
    main(device_id)