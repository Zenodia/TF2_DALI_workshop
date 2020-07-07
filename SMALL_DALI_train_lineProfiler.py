import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
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
       


    def define_graph(self):
        imgs, labels = self.input_img()
        images = self.decode(imgs)
        return (images, labels)

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
def get_pipeline(device_id,batch_size):
    pipe = Opt3Pipeline("smallimgs_path.txt", 2, 0, batch_size, 8, device_id)
    return pipe
@profile
def main(device_id, batch_size):
    with tf.device('/gpu:{}'.format(str(device_id))):

        pipe=get_pipeline(device_id,batch_size)
        # Create dataset
        # Define shapes and types of the outputs
        shapes = [
            (batch_size, 224, 224,3),
            (batch_size,1)]
        dtypes = [
            tf.float32,
            tf.int32]
        out = dali_tf.DALIDataset(
                pipeline=pipe,
                batch_size=batch_size,
                shapes=shapes,
                dtypes=dtypes,
                device_id=0)


        #out = out.with_options(dataset_options())

        m=get_reset50_model()
        m.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # Train using DALI dataset
        m.fit(
            out,
            epochs=10,
            steps_per_epoch=batch_size*8,
            use_multiprocessing=False
        )
if __name__ == "__main__":    
    label2num={'notHappy':0,'Happy':1,'others':2}
    num2label=dict([(b,a) for (a,b) in label2num.items()])
    device_id=0
    batch_size=8
    main(device_id,batch_size)
