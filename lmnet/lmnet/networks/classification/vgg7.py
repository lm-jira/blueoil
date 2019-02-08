# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from functools import partial

import tensorflow as tf

from lmnet.networks.classification.base import Base

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg7Network(Base):

    def __init__(
            self,
            weight_decay_rate=None,
            classes=(),
            optimizer_class=tf.train.MomentumOptimizer,
            optimizer_kwargs=None,
            is_debug=False,
            learning_rate_func=None,
            learning_rate_kwargs=None,
            image_size=(),  # [height, width]
            batch_size=64,
            data_format='NHWC',
            *args,
            **kwargs
    ):

        self.num_classes = len(classes)
        self.data_format = data_format
        self.custom_getter = None
        self.activation = tf.nn.relu

        super().__init__(
            weight_decay_rate,
            is_debug,
            optimizer_class,
            optimizer_kwargs,
            learning_rate_func,
            learning_rate_kwargs,
            classes,
            image_size,
            batch_size,
            data_format,
        )

    def base(self, images, is_training):
        self.images = images

        self.input = self.convert_rgb_to_bgr(images)
        self.input = tf.cast(self.input, dtype=tf.float32)
        self.input = tf.div(self.input, 255)

        self.conv1 = self.conv_layer("conv1", self.input, filters=128, kernel_size=3, is_training=is_training)
        self.conv2 = self.conv_layer("conv2", self.conv1, filters=128, kernel_size=3, is_training=is_training)

        self.pool1 = self.max_pool("pool1", self.conv2, kernel_size=2, strides=2)

        self.conv3 = self.conv_layer("conv3", self.pool1, filters=256, kernel_size=3, is_training=is_training)
        self.conv4 = self.conv_layer("conv4", self.conv3, filters=256, kernel_size=3, is_training=is_training)
        
        self.pool2 = self.max_pool("pool2", self.conv4, kernel_size=2, strides=2)
    
        self.conv5 = self.conv_layer("conv5", self.pool2, filters=512, kernel_size=3, is_training=is_training)
        self.conv6 = self.conv_layer("conv6", self.conv5, filters=512, kernel_size=3, is_training=is_training)
        
        self.pool3 = self.max_pool("pool3", self.conv6, kernel_size=2, strides=2)

        self.flatten = tf.contrib.layers.flatten(self.pool3)
        
        self.fc1 = self.fc_layer("fc1", self.flatten, filters=1024, activation=None)
        self.batch_normed1 = tf.contrib.layers.batch_norm(self.fc1,
                                                          epsilon=0.00001,
                                                          scale=True,
                                                          center=True,
                                                          updates_collections=None,
                                                          is_training=is_training,
                                                          data_format=self.data_format)
        
        self.activated1 = tf.nn.relu(self.batch_normed1)
        self.fc2 = self.fc_layer("fc2", self.activated1, filters=self.num_classes, activation=None)
        
        return self.fc2

    def conv_layer(
            self,
            name,
            inputs,
            filters,
            kernel_size,
            is_training=True,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            *args,
            **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        if self.data_format == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False,
                                    padding=padding,
                                    strides=strides,
                                    data_format=data_format,
            )

        batch_normed = tf.contrib.layers.batch_norm(
            conv,
            epsilon=0.00001,
            scale=True,
            center=True,
            updates_collections=None,
            is_training=is_training,
            data_format=self.data_format
        )

        output = tf.nn.relu(batch_normed)

        return output

    def fc_layer(
            self,
            name,
            inputs,
            filters,
            activation,
            biases_initializer=tf.zeros_initializer(),
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(name):
            output = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=filters,
                weights_initializer=kernel_initializer,
                biases_initializer=biases_initializer,
                activation_fn=activation,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)
            )
            
        return output
        
    def convert_rgb_to_bgr(self, rgb_images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_images)
            
        bgr_images = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr_images

    def max_pool(
            self,
            name,
            inputs,
            kernel_size=2,
            strides=2
    ):
        if self.data_format == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

        output = tf.layers.max_pooling2d(
            name=name,
            inputs=inputs,
            pool_size=kernel_size,
            strides=strides,
            data_format=data_format,
        )
        
        return output

    def loss(self, softmax, labels):
        loss = super().loss(softmax, labels)

        fc_regularized_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += fc_regularized_loss

        return loss

    
class Vgg7Quantize(Vgg7Network):
    """VGG7 quantize network for classification, version 1.0

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
    """
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
            if "weights" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
