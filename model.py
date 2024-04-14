import Cnn
import tensorflow as tf
import numpy as np

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
    def __init__(self,hps):
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
        super(CnnEncoder,self).build(input_shape)

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
        super(CnnDecoder,self).build(input_shape)

    def call(self,inputs,training=None):
        fc1 = self.fc_net(inputs,training)
        fc1 = tf.reshape(fc1,shape=[-1,3,3,256])
        return self.conv_net(fc1,training)

class RnnDecoder(tf.keras.layers.Layer):
    def __init__(self,cell,hps):
        super(RnnDecoder,self).__init__(name='decoder')
        self.cell = cell
        self.hps = hps
        self.n_out = 3 + 20 * 6
        self.fc_spec = [('linear',self.n_out,'fc')]
        self.fc_net = Cnn.FcNet(self.fc_spec)
        
    
    def build(self,input_shape):
        self.rnn_layer = tf.keras.layers.RNN(
            self.cell, 
            return_sequences=True, 
            return_state=True,
            name='rnn_output'
        )
        super(RnnDecoder,self).build(input_shape)
    
    def call(self,inputs,initial_state,training=None):
        self.output, last_state = self.rnn_layer(inputs, initial_state=initial_state)
        self.output = self.fc_net(self.output,training)
        out = self.get_mixture_params(self.output)
        last_state = tf.identity(self.last_state, name='last_state')
        return out, last_state

    def get_mixture_params(self, output):
        pen_logits = output[:, 0:3]
        pi, mu1, mu2, sigma1, sigma2, corr = tf.split(output[:, 3:], 6, 1)

        pi = tf.nn.softmax(pi)
        pen = tf.nn.softmax(pen_logits)

        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        corr = tf.tanh(corr)

        r = [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
        return r

class Model(tf.keras.Model):
    def __init__(self, hps):
        super(Model, self).__init__()
        self.hps = hps
        self.k = self.hps.num_mixture * self.hps.num_sub  
        
        # Gaussian number
        self.add_variables()
    
    def add_variables(self):
        self.global_ = self.add_weight(
            name='global_',
            shape=[],
            initializer=tf.ones_initializer(dtype=tf.float32),
            trainable=False)
        self.de_mu = self.add_weight(
            name='de_mu',
            shape=[self.k, self.hps.z_size],
            initializer=tf.random.uniform_initializer(minval=-1., maxval=1.),
            trainable=False)
        self.de_sigma2 = self.add_weight(
            name='de_sigma2',
            shape=[self.k, self.hps.z_size],
            initializer=tf.ones_initializer(),
            trainable=False)
        self.de_alpha = self.add_weight(
            name='de_alpha',
            shape=[self.k, 1],
            initializer=tf.constant_initializer(1. / float(self.k)),
            trainable=False)

    def get_decoder_cell(self):
        # Decoder cell configuration
        if self.hps.dec_model == 'lstm':
            return tf.keras.layers.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            return tf.keras.layers.LayerNormalization
        else:
            raise ValueError('Please choose a respectable cell')
        
    def setup_optimizer(self,loss):
        self.lr = (self.hps.learning_rate - self.hps.min_learning_rate) * \
                  (self.hps.decay_rate ** (self.global_ / 3)) + self.hps.min_learning_rate
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        gvs = optimizer.compute_gradients(loss)
        g= self.hps.grad_clip
        clipped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
        return optimizer , clipped_gvs
    
    def build(self, input_shape, training=None):
        self.inputseqs , self.input_pngs = input_shape
        self.input_x,self.output_x = self.prepare_inputs(self.inputseqs,self.input_pngs)
        self.target = tf.reshape(self.output_x,[-1,5])
        self.x1_data, self.x2_data, self.pen_data = tf.split(self.target, [1, 1, 3], axis=1)

        self.encoder = CnnEncoder(self.hps)
        self.decoder = CnnDecoder()
        self.cell = self.get_decoder_cell()
        fc_spec = [('tanh',self.cell.state_size,'init_state')]
        self.fc_net = Cnn.FcNet(fc_spec)
        self.rnn_decoder = RnnDecoder(self.cell,self.hps)
        if self.hps.use_input_dropout:
            print(f'Dropout to input w/ keep_prob = {self.hps.input_dropout_prob}.')
            self.input_dropout = tf.keras.layers.Dropout(rate=(1 - self.hps.input_dropout_prob))
        else:
            self.input_dropout = None

        if self.hps.use_output_dropout:
            print(f'Dropout to output w/ keep_prob = {self.hps.output_dropout_prob}.')
            self.output_dropout = tf.keras.layers.Dropout(rate=(1 - self.hps.output_dropout_prob))
        else:
            self.output_dropout = None
        super(Model,self).build(input_shape)
    def call(self, training=None):
        # CNN encoder
        self.p_mu, self.p_sigma2 = self.encoder(self.input_pngs,self.hps,training=training)
        self.batch_z = self.get_z(self.p_mu, self.p_sigma2)


        # Compute decoder initial state and input
        self.initial_state = self.fc_net(self.batch_z,training)
        pre_z = tf.tile(tf.reshape(self.batch_z, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.hps.max_seq_len, 1])
        dec_input = tf.concat([self.input_x, pre_z], axis=2)

        self.gen_img = self.decoder(dec_input,training)
        # Generation branch
        self.dec_out, self.last_state = self.rnn_decoder(dec_input,self.initial_state,training)
        self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.pen, self.pen_logits = self.dec_out

        self.p_alpha, self.gau_label = self.calculate_posterior(self.batch_z, self.de_mu, self.de_sigma2, self.de_alpha)

        self.q_mu, self.q_sigma2, self.q_alpha \
            = tf.cond(tf.cast(self.global_ > 500, tf.bool),
                      lambda: tf.cond(tf.cast(self.global_ % 3, tf.bool),
                                      lambda: self.em(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha),
                                      lambda: self.rpcl(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha)),
                      lambda: self.em(self.p_mu, self.p_sigma2, self.p_alpha, self.de_mu, self.de_sigma2, self.de_alpha))

        # Loss function
        self.alpha_loss = self.get_alpha_loss(self.p_alpha, tf.stop_gradient(self.q_alpha))
        self.gaussian_loss = self.get_gaussian_loss(self.p_alpha, self.p_mu, self.p_sigma2, tf.stop_gradient(self.q_mu), tf.stop_gradient(self.q_sigma2))
        self.lil_loss = self.get_lil_loss(self.pi, self.mux, self.muy, self.sigmax, self.sigmay, self.corr,
                                          self.pen_logits, self.x1_data, self.x2_data, self.pen_data)
        self.de_loss = self.calculate_deconv_loss(self.input_pngs, self.gen_img, 'square')

        self.kl_weight = 1. - 0.999 * (0.9999 ** self.global_)  # Warm up
        self.loss = self.kl_weight * (self.alpha_loss + self.gaussian_loss) + self.lil_loss + self.hps.de_weight * self.de_loss

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        gvs = optimizer.compute_gradients(self.loss)
        g = self.hps.grad_clip
        for i, (grad, var) in enumerate(gvs):
            if grad is not None:
                gvs[i] = (tf.clip_by_value(grad, -g, g), var)
        self.train_op = optimizer.apply_gradients(gvs)

        # Update the GMM parameters
        self.update_gmm_mu = tf.assign(self.de_mu, self.q_mu)
        self.update_gmm_sigma2 = tf.assign(self.de_sigma2, self.q_sigma2)
        self.update_gmm_alpha = tf.assign(self.de_alpha, self.q_alpha)
    
    def get_z(self, mu, sigma2):
        """ Reparameterization """
        sigma = tf.sqrt(sigma2)
        eps = tf.random.normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        z = tf.add(mu, tf.multiply(sigma, eps), name='z_code')
        return z
    
    def prepare_inputs(self, input_seqs, input_pngs):
        input_x = input_seqs[:, :self.hps.max_seq_len, :]
        output_x = input_seqs[:, 1:self.hps.max_seq_len + 1, :]
        return input_x, output_x
    
    def calculate_prob(self, x, q_mu, q_sigma2):
        """ Calculate the probabilistic density """
        mu = tf.tile(tf.reshape(q_mu, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        sigma2 = tf.tile(tf.reshape(q_sigma2, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        x = tf.tile(tf.reshape(x, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        log_exp_part = -0.5 * tf.reduce_sum(tf.divide(tf.square(x - mu), 1e-30 + sigma2), axis=2)
        log_frac_part = tf.reduce_sum(tf.log(tf.sqrt(sigma2 + 1e-30)), axis=2)
        log_prob = log_exp_part - log_frac_part - float(self.hps.z_size) / 2. * tf.log(2. * 3.1416)
        return tf.exp(tf.to_double(log_prob))

    def calculate_posterior(self, y, q_mu, q_sigma2, q_alpha):
        """ Calculate the posterior p(y|k) """
        prob = self.calculate_prob(y, q_mu, q_sigma2)
        temp = tf.multiply(tf.tile(tf.transpose(tf.to_double(q_alpha), [1, 0]), [self.hps.batch_size, 1]), prob)
        sum_temp = tf.tile(tf.reduce_sum(temp, axis=1, keep_dims=True), [1, self.k])
        gamma = tf.clip_by_value(tf.to_float(tf.divide(temp, 1e-300 + sum_temp)), 1e-5, 1.)
        gamma_st = gamma / (1e-10 + tf.tile(tf.reduce_sum(gamma, axis=1, keep_dims=True), [1, self.k]))
        return gamma_st, tf.argmax(gamma_st, axis=1)

    def rpcl(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        """ EM-like algorithm enhanced by RPCL """
        en_sigma2 = tf.tile(tf.reshape(en_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        with tf.name_scope('rpcl'):
            penalize = 1e-4  # De-learning rate
            temp_y = tf.tile(tf.expand_dims(y, axis=1), [1, self.k, 1])
            winner = tf.one_hot(tf.argmax(gamma, axis=1), self.k, axis=1)  # the winner
            rival = tf.one_hot(tf.argmax(gamma - gamma * winner, axis=1), self.k, axis=1)  # the rival
            gamma_rpcl = winner - penalize * rival
            sum_gamma_rpcl = tf.tile(tf.expand_dims(tf.reduce_sum(gamma_rpcl, axis=0), axis=1), [1, self.hps.z_size])
            q_mu_new = tf.reduce_sum(temp_y * tf.tile(tf.expand_dims(gamma_rpcl, axis=2),
                                                      [1, 1, self.hps.z_size]), axis=0) / (sum_gamma_rpcl + 1e-10)
            q_sigma2_new = tf.reduce_sum((tf.square(temp_y - tf.tile(tf.expand_dims(q_mu_new, axis=0), [self.hps.batch_size, 1, 1])) + en_sigma2)
                                         * tf.tile(tf.expand_dims(gamma_rpcl, axis=2), [1, 1, self.hps.z_size]), axis=0) \
                           / (sum_gamma_rpcl + 1e-10)
            q_alpha_new = tf.expand_dims(tf.reduce_mean(gamma_rpcl, axis=0), axis=1)

            q_mu = q_mu_old * 0.95 + q_mu_new * 0.05
            q_sigma2 = tf.clip_by_value(q_sigma2_old * 0.95 + q_sigma2_new * 0.05, 1e-10, 1e10)
            q_alpha = tf.clip_by_value(q_alpha_old * 0.95 + q_alpha_new * 0.05, 0., 1.)
            q_alpha_st = q_alpha / tf.reduce_sum(q_alpha)

            return q_mu, q_sigma2, q_alpha_st

    def em(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        """ EM algorithm for GMM learning """
        en_sigma2 = tf.tile(tf.reshape(en_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])

        with tf.name_scope('em'):
            sum_gamma = tf.tile(tf.expand_dims(tf.reduce_sum(gamma, axis=0), axis=1), [1, self.hps.z_size])
            temp_y = tf.tile(tf.expand_dims(y, axis=1), [1, self.k, 1])

            q_mu_new = tf.reduce_sum(temp_y * tf.tile(tf.expand_dims(gamma, axis=2), [1, 1, self.hps.z_size]), axis=0) / (sum_gamma + 1e-10)
            q_sigma2_new = tf.reduce_sum((tf.square(temp_y - tf.tile(tf.expand_dims(q_mu_new, axis=0), [self.hps.batch_size, 1, 1])) + en_sigma2)
                                         * tf.tile(tf.expand_dims(gamma, axis=2), [1, 1, self.hps.z_size]), axis=0) / (sum_gamma + 1e-10)
            q_alpha_new = tf.expand_dims(tf.reduce_mean(gamma, axis=0), axis=1)

            q_mu = q_mu_old * 0.95 + q_mu_new * 0.05
            q_sigma2 = tf.clip_by_value(q_sigma2_old * 0.95 + q_sigma2_new * 0.05, 1e-10, 1e10)
            q_alpha = tf.clip_by_value(q_alpha_old * 0.95 + q_alpha_new * 0.05, 0., 1.)
            q_alpha_st = q_alpha / tf.reduce_sum(q_alpha)

            return q_mu, q_sigma2, q_alpha_st
        
    def get_alpha_loss(self, p_alpha, q_alpha):
        p_alpha = tf.reshape(p_alpha, [self.hps.batch_size, self.k])
        q_alpha = tf.tile(tf.reshape(q_alpha, [1, self.k]), [self.hps.batch_size, 1])
        return tf.reduce_sum(tf.reduce_mean(p_alpha * tf.log(tf.div(p_alpha, q_alpha + 1e-10) + 1e-10), axis=0))

    def get_gaussian_loss(self, p_alpha, p_mu, p_sigma2, q_mu, q_sigma2):
        p_alpha = tf.reshape(p_alpha, [self.hps.batch_size, self.k])
        p_mu = tf.tile(tf.reshape(p_mu, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])
        p_sigma2 = tf.tile(tf.reshape(p_sigma2, [self.hps.batch_size, 1, self.hps.z_size]), [1, self.k, 1])
        q_mu = tf.tile(tf.reshape(q_mu, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        q_sigma2 = tf.tile(tf.reshape(q_sigma2, [1, self.k, self.hps.z_size]), [self.hps.batch_size, 1, 1])
        return tf.reduce_mean(tf.reduce_sum(0.5 * tf.multiply(p_alpha, tf.reduce_sum(
            tf.log(q_sigma2 + 1e-10) + tf.div(p_sigma2 + (p_mu - q_mu) ** 2, q_sigma2 + 1e-10) - 1.0 - tf.log(
                p_sigma2 + 1e-10), axis=2)), axis=1))

    def calculate_deconv_loss(self, img, gen_img, sign):
        img = tf.reshape(img, [self.hps.batch_size, self.hps.png_width ** 2])
        gen_img = tf.reshape(gen_img, [self.hps.batch_size, self.hps.png_width ** 2])
        if sign == 'square':
            return tf.reduce_mean(tf.reduce_sum(tf.square(img - gen_img), axis=1))
        elif sign == 'absolute':
            return tf.reduce_mean(tf.reduce_sum(tf.abs(img - gen_img), axis=1))
        else:
            assert False, 'please choose a respectable cell'

    def get_density(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
            2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result

    def get_lil_loss(self, pi, mu1, mu2, s1, s2, corr, pen_logits, x1_data, x2_data, pen_data):
        result0 = self.get_density(x1_data, x2_data, mu1, mu2, s1, s2, corr)
        epsilon = 1e-6
        result1 = tf.multiply(result0, pi)
        result1 = tf.reduce_sum(result1, axis=1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # Avoid log(0)

        masks = 1.0 - pen_data[:, 2]
        masks = tf.reshape(masks, [-1, 1])
        result1 = tf.multiply(result1, masks)

        result2 = tf.nn.softmax_cross_entropy_with_logits(logits=pen_logits, labels=pen_data)
        result2 = tf.reshape(result2, [-1, 1])

        if not self.hps.is_training:
            result2 = tf.multiply(result2, masks)
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(result1 + result2, [self.hps.batch_size, -1]), axis=1))
    