#   Copyright 2021 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   P.S. We thank Ha and Eck [1] for their public source codes.
#        And the details about their work can be found below.
#
#       [1] https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn
#
""" Modules for CNN & FC layers"""

import tensorflow as tf
import tensorflow_addons as tfa


class DilatedConv(tf.keras.layers.Layer):
    def __init__(self, specs, keep_prob=1.0, **kwargs):
        super(DilatedConv, self).__init__(**kwargs)
        self.specs = specs 
        self.keep_prob = keep_prob
        self.conv_layers = []

    def build(self, input_shape):
        for i,(fun_name, w_size, rate, out_channel) in enumerate(self.specs):
            #with tf.name_scope('conv%d' % i):
            conv_layer = tf.keras.layers.Conv2D(
                filters=out_channel,
                kernel_size=w_size,
                padding='SAME',
                dilation_rate=rate,
                activation=None,
                use_bias=False,
                kernel_initializer='glorot_uniform',
                input_shape=input_shape if i == 0 else None,
                #name='conv%d' % i
            )
            bn_layer = tf.keras.layers.BatchNormalization()
            self.conv_layers.append((conv_layer, bn_layer, fun_name))

    def call(self, inputs, training=None):
        x = inputs
        for conv_layer, bn_layer, fun_name in self.conv_layers:
            x = conv_layer(x, training=training)
            x = bn_layer(x, training=training)
            x = tf.keras.activations.get(fun_name)(x)
        return x

class ConvNet(tf.keras.layers.Layer):
    def __init__(self, specs, keep_prob=1.0, deconv=False,**kwargs):
        super(ConvNet, self).__init__(**kwargs)
        self.specs = specs
        self.keep_prob = keep_prob
        self.conv_layers = []
        self.deconv = deconv

    def build(self,input_shape):
        for i, (fun_name, w_size, strides, out_channel) in enumerate(self.specs):
            if not self.deconv:
            #with tf.name_scope('conv%d' % i):
                conv_layer = tf.keras.layers.Conv2D(
                    filters=out_channel,
                    kernel_size=w_size,
                    strides=strides,
                    padding='SAME',
                    activation=None,
                    use_bias=False,
                    kernel_initializer='glorot_uniform',
                    input_shape=input_shape if i == 0 else None,
                    #name= 'conv%d' % i
                )
            else:
            #with tf.name_scope('deconv%d' % i):
                conv_layer = tf.keras.layers.Conv2DTranspose(
                    filters=out_channel,
                    kernel_size=w_size,
                    strides=strides,
                    padding='SAME',
                    activation=None,
                    use_bias=False,
                    kernel_initializer='glorot_uniform',
                    input_shape=input_shape if i == 0 else None,
                    #name='deconv%d' % i
                )
        bn_layer = tf.keras.layers.BatchNormalization()
        self.conv_layers.append((conv_layer, bn_layer, fun_name))


    def call(self, inputs, training=None):
        x = inputs
        for conv_layer, bn_layer, fun_name in self.conv_layers:
            x = conv_layer(x, training=training)
            x = bn_layer(x, training=training)
            x = tf.keras.activations.get(fun_name)(x)
        return x

class FcNet(tf.keras.layers.Layer):
    def __init__(self, specs, **kwargs):
        super(FcNet, self).__init__(**kwargs)
        self.specs = specs
        self.fc_layers = []
    
    def build(self, input_shape):
        for i, (fun_name, out_channel,name) in enumerate(self.specs):
            #with tf.name_scope(name):
            fc_layer = tf.keras.layers.Dense(
                units=out_channel,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                input_shape=input_shape if i == 0 else None,
                #name=name
            )
            self.fc_layers.append((fc_layer, fun_name))


    def call(self, inputs, training=None):
        x = inputs
        for fc_layer, fun_name in self.fc_layers:
            x = fc_layer(x, training=training)
            x = tf.keras.activations.get(fun_name)(x)
        return x
