import Cnn
import tensorflow as tf

class DcnLayer(tf.keras.layers.Layer):
    def __init__(self,depth,name=None):
        super().__init__(name=name)
        self.conv = Cnn.DilatedConv([('linear',(3,3),1,depth)])
    
    def call(self,inputs,training=None):
        return self.conv(inputs,training)
    
class CombineLayer(tf.keras.layers.Layer):
    def __init__(self,depth,stride,name=None):
        super().__init__(name=name)
        self.conv = Cnn.ConvNet([('linear',(3,3),(stride,stride),depth)])
    
    def call(self,dcn1_out,dcn2_out,dcn3_out,training=None):
        combine_out = tf.nn.relu(dcn1_out + dcn2_out + dcn3_out)
        return self.conv(combine_out,training)


class DilatedBlocks(tf.keras.layers.Layer):
    def __init__(self,depth,name=None,stride=1,**kwargs):
        super().__init__(name=name,**kwargs)
        self.dcn1 = DcnLayer(depth,'dcn1')
        self.dcn2 = DcnLayer(depth,'dcn2')
        self.dcn3 = DcnLayer(depth,'dcn3')
        self.combine = CombineLayer(depth,stride,'combine')
    def call(self,inputs,training=None):
        dcn1_out = self.dcn1(inputs,training)
        dcn2_out = self.dcn2(inputs,training)
        dcn3_out = self.dcn3(inputs,training)
        combine_out = self.combine(dcn1_out,dcn2_out,dcn3_out,training)
        return tf.nn.relu(combine_out)
    
class CnnEncoder(tf.keras.layers.Layer):
    def __init__(self,inputs,hps,training=None):
        super().__init__(name='encoder')
        self.dilated_block_specs = [
                (32, 'layer_1',2),
                (64, 'layer_2',2),
            ]
        self.conv_specs = [
                    ('relu', (3, 3), (2, 2), 128),
                    ('relu', (3, 3), (2, 2), 256),
                ]
        self.fc_spec_mu = [('linear', hps.z_size, 'fc_mu')]
        self.fc_spec_sigma2 = [('linear', hps.z_size, 'fc_sigma2')]
    
    def build(self,input_shape):
        self.dilated_blocks = [DilatedBlocks(depth,name,stride) for depth,name,stride in self.dilated_block_specs]
        self.conv_net = Cnn.ConvNet(self.conv_specs)
        self.fc_net_mu = Cnn.FcNet(self.fc_spec_mu)
        self.fc_net_sigma2 = Cnn.FcNet(self.fc_spec_sigma2)

    def call(self,inputs,training=None):
        x = inputs
        for block in self.dilated_blocks:
            x = block(x,training)

        conv_out = self.conv_net(x,training)
        conv_out_reshaped = tf.reshape(conv_out,shape=[-1,3 * 3 * 256])

        p_mu = self.fc_net_mu(conv_out,training)
        p_sigma2 = self.fc_net_sigma2(conv_out_reshaped,training)
        return p_mu, tf.nn.softplus(p_sigma2) + 1e-10

    
class CnnDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(CnnDecoder,self).__init__(name='deconv')
        self.fc_spec = [('relu',3 * 3 * 256, 'fc1')]
        self.de_conv_specs = [
            ('relu', (3, 3), (2,2), 128),
            ('relu', (3, 3), (2,2), 64),
            ('relu', (3, 3), (2,2), 32),
            ('tanh', (3, 3), (2,2), 1)
        ]

    def build(self,input_shape):
        # maybe we should use the input_shape to build the fc_net and conv_net?
        self.fc_net = Cnn.FcNet(self.fc_spec)
        self.conv_net = Cnn.ConvNet(self.de_conv_specs,deconv=True)

    def call(self,inputs,training=None):
        fc1 = self.fc_net(inputs,training)
        fc1 = tf.reshape(fc1,shape=[-1,3,3,256])
        return self.conv_net(fc1,training)
    

class Model(tf.keras.Model):
    def __init__(self, hps):
        super(Model, self).__init__()
        self.hps = hps
        self.k = self.hps.num_mixture * self.hps.num_sub  # Gaussian number
        self.global_ = tf.Variable(initial_value=tf.ones(shape=[], dtype=tf.float32), trainable=False)
        self.de_mu = tf.Variable(initial_value=tf.random.uniform(shape=[self.k, self.hps.z_size], minval=-1., maxval=1.), trainable=False)
        self.de_sigma2 = tf.Variable(initial_value=tf.ones(shape=[self.k, self.hps.z_size]), trainable=False)
        self.de_alpha = tf.Variable(initial_value=tf.constant(1. / float(self.k), shape=[self.k, 1], dtype=tf.float32), trainable=False)

        # Decoder cell configuration
        if self.hps.dec_model == 'lstm':
            cell_fn = tf.keras.layers.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = tf.keras.layers.LayerNormalization
        else:
            raise ValueError('Please choose a respectable cell')

        cell = cell_fn(self.hps.dec_rnn_size)

        if hps.use_input_dropout:
            print(f'Dropout to input w/ keep_prob = {hps.input_dropout_prob}.')
            self.input_dropout = tf.keras.layers.Dropout(rate=(1 - hps.input_dropout_prob))
        else:
            self.input_dropout = None

        if hps.use_output_dropout:
            print(f'Dropout to output w/ keep_prob = {hps.output_dropout_prob}.')
            self.output_dropout = tf.keras.layers.Dropout(rate=(1 - hps.output_dropout_prob))
        else:
            self.output_dropout = None

        self.rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
        
    def call(self, inputs, training=None):
        # Decoder RNN example, assuming 'inputs' is provided via tf.data.Dataset API
        input_seqs, input_pngs = inputs
    
    def prepare_inputs(self, input_seqs, input_pngs):
        input_x = input_seqs[:, :self.hps.max_seq_len, :]
        output_x = input_seqs[:, 1:self.hps.max_seq_len + 1, :]
        return input_x, output_x
    
    def train_step(self, data):
        input_seqs, input_pngs = data
        input_x,output_x = self.prepare_inputs(input_seqs, input_pngs)

        with tf.GradiateTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss}

# Instantiate and compile the model, for example:
hps = ...  # define your hyperparameters
model = Model(hps)
model.compile(optimizer='adam', loss='...')  # define loss and optimizer as per your requirements