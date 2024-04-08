import Cnn
import tensorflow as tf

class DcnLayer(tf.Module):
    def __init__(self,depth,name=None):
        super().__init__(name)
        with tf.name_scope(name):
            self.conv = Cnn.DilatedConv([('linear',(3,3),1,depth)])
    
    def __call__(self,inputs,training=None):
        return self.conv(inputs,training)
    
class CombineLayer(tf.Module):
    def __init__(self,depth,stride,name=None):
        super().__init__(name)
        with tf.name_scope(name):
            self.conv = Cnn.ConvNet([('linear',(3,3),(stride,stride),depth)])
    
    def __call__(self,dcn1_out,dcn2_out,dcn3_out,training=None):
        combine_out = tf.nn.relu(dcn1_out + dcn2_out + dcn3_out)
        return self.conv(combine_out,training)


class DilatedBlocks(tf.Module):
    def __init__(self,depth,name=None,stride=1):
        super().__init__(name)
        with tf.name_scope(name):
            self.dcn1 = DcnLayer(depth,'dcn1')
            self.dcn2 = DcnLayer(depth,'dcn2')
            self.dcn3 = DcnLayer(depth,'dcn3')
            self.combine = CombineLayer(depth,stride,'combine')
    def __call__(self,inputs,training=None):
        dcn1_out = self.dcn1(inputs,training)
        dcn2_out = self.dcn2(inputs,training)
        dcn3_out = self.dcn3(inputs,training)
        return tf.nn.relu(self.combine(dcn1_out,dcn2_out,dcn3_out,training))