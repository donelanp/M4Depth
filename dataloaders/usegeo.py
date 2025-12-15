import tensorflow as tf

from .generic import *

class DataLoaderUseGeo(DataLoaderGeneric):
    def __init__(self):
        super(DataLoaderUseGeo, self).__init__("usegeo")
        self.depth_type = "map"

    def _perform_augmentation(self):
        self._augmentation_step_color(invert_color=False)

    def _set_output_size(self, out_size=[384, 384]):
        self.out_size = out_size

    @tf.function
    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera']], separator='/'))
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        cx = data_sample['c'] * self.out_size[1] / 1320.0
        cy = data_sample['c'] * self.out_size[0] / 1989.0
        fx = data_sample['fx'] * self.out_size[1] / 1320.0
        fy = data_sample['fy'] * self.out_size[0] / 1989.0

        camera_data = {
            "f": tf.convert_to_tensor([fx, fy], dtype=tf.float32),
            "c": tf.convert_to_tensor([cx, cy], dtype=tf.float32),
        }
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.out_size), self.out_size+[3])
        out_data['rot'] = tf.cast(tf.stack([data_sample['qw'],data_sample['qx'],data_sample['qy'],data_sample['qz']], 0), dtype=tf.float32)
        out_data['trans'] = tf.cast(tf.stack([data_sample['tx'],data_sample['ty'],data_sample['tz']], 0), dtype=tf.float32)
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)

        # Load depth data only if they are available
        if 'depth' in data_sample:
            file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
            image = tf.image.decode_png(file, dtype=tf.uint16)
            depth = tf.cast(image, dtype=tf.float32)/256
            out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])

        return out_data
