import os
import tensorflow as tf
from BasicConvLSTMCell import BasicConvLSTMCell
from operations import *
from utils import *


class NETWORK(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=10, T=10, checkpoint_dir=None, is_train=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train

        self.gf_dim = 64
        self.df_dim = 64

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.diff_shape = [batch_size, self.image_size[0],
                           self.image_size[1], K - 1, 1]
        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             K + T, c_dim]

        self.build_model()

    def build_model(self):
        self.diff_in = tf.placeholder(tf.float32, self.diff_shape, name='diff_in')
        self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')

        cell = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], 256)
        pred = self.forward(self.diff_in, self.xt, cell)

        self.G = tf.concat(axis=3, values=pred)
        if self.is_train:
            true_sim = inverse_transform(self.target[:, :, :, self.K:, :])
            if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
                                  [-1, self.image_size[0],
                                   self.image_size[1], 3])
            gen_sim = inverse_transform(self.G)
            if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
                                 [-1, self.image_size[0],
                                  self.image_size[1], 3])
            binput = tf.reshape(self.target[:, :, :, :self.K, :],
                                [self.batch_size, self.image_size[0],
                                 self.image_size[1], -1])
            btarget = tf.reshape(self.target[:, :, :, self.K:, :],
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])

            target_data = tf.concat(axis=3, values=[binput, btarget])
            gen_data = tf.concat(axis=3, values=[binput, bgen])

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(target_data)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data)

            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target[:, :, :, self.K:, :])
            )
            self.L_gdl = gdl(gen_sim, true_sim, 1.)
            self.L_img = self.L_p + self.L_gdl

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.L_GAN = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.ones_like(self.D_)
                )
            )

            self.loss_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()));
            print("Number of parameters: %d" % num_param)
        self.saver = tf.train.Saver(max_to_keep=10)

    def forward(self, diff_in, xt, cell):
        # Initial state
        state = tf.zeros([self.batch_size, self.image_size[0] / 8,
                          self.image_size[1] / 8, 512])
        reuse = False
        # Encoder
        for t in xrange(self.K - 1):
            enc_h, res_encoder = self.encoder_vgg(diff_in[:, :, :, t, :], reuse=reuse)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse)
            reuse = True

        pred = []
        # Decoder
        for t in xrange(self.T):
            if t == 0:
                diff_hat = self.decoder_vgg(h_dyn, res_encoder, reuse=False)
            else:
                enc_h, res_encoder = self.encoder_vgg(diff_in, reuse=True)
                h_dyn, state = cell(enc_h, state, scope='lstm', reuse=True)
                diff_hat = self.decoder_vgg(h_dyn, res_encoder, reuse=True)

            x_hat = diff_hat + xt

            diff_in = inverse_transform(diff_hat)
            xt = x_hat
            pred.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                           self.image_size[1], 1, self.c_dim]))

        return pred



    def encoder_vgg(self, diff_in, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv1_1', reuse=reuse))
        conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv1_2', reuse=reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv2_1', reuse=reuse))
        conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv2_2', reuse=reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv3_1', reuse=reuse))
        conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv3_2', reuse=reuse))
        conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv3_3', reuse=reuse))

        res_in.append(conv3_3)

        conv4_1 = relu(conv2d(conv3_3, output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv4_1', reuse=reuse))
        conv4_2 = relu(conv2d(conv4_1 , output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv4_2', reuse=reuse))
        conv4_3 = relu(conv2d(conv4_2, output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv4_3', reuse=reuse))
        res_in.append(conv4_3)

        conv5_1 = relu(conv2d(conv4_3, output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv5_1', reuse=reuse))
        conv5_2 = relu(conv2d(conv5_1, output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv5_2', reuse=reuse))
        conv5_3 = relu(conv2d(conv5_2, output_dim=self.gf_dim * 8, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='encoder_vgg_conv5_3', reuse=reuse))
        res_in.append(conv5_3)
        pool5 = MaxPooling(conv4_3, [2, 2])

        return pool5, res_in


    def decoder_vgg(self, h_comb, res_connect, reuse=False):

        shapel5 = [self.batch_size, self.image_size[0] / 4,
                   self.image_size[1] / 4, self.gf_dim * 8]
        shapeout5 = [self.batch_size, self.image_size[0] / 4,
                     self.image_size[1] / 4, self.gf_dim * 8]
        depool5 = FixedUnPooling(h_comb, [2, 2])
        deconv5_3 = relu(deconv2d(relu(tf.add(depool5, res_connect[4])),
                                  output_shape=shapel5, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv5_3', reuse=reuse))
        deconv5_2 = relu(deconv2d(deconv5_3, output_shape=shapel5, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv5_2', reuse=reuse))
        deconv5_1 = relu(deconv2d(deconv5_2, output_shape=shapeout5, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv5_1', reuse=reuse))

        shapel4 = [self.batch_size, self.image_size[0] / 4,
                   self.image_size[1] / 4, self.gf_dim * 8]
        shapeout4 = [self.batch_size, self.image_size[0] / 4,
                     self.image_size[1] / 4, self.gf_dim * 4]

        deconv4_3 = relu(deconv2d(relu(tf.add(deconv5_1, res_connect[3])),
                                  output_shape=shapel4, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv4_3', reuse=reuse))
        deconv4_2 = relu(deconv2d(deconv4_3, output_shape=shapel4, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv4_2', reuse=reuse))
        deconv4_1 = relu(deconv2d(deconv4_2, output_shape=shapeout4, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv4_1', reuse=reuse))

        shapel3 = [self.batch_size, self.image_size[0] / 4,
                   self.image_size[1] / 4, self.gf_dim * 4]
        shapeout3 = [self.batch_size, self.image_size[0] / 4,
                     self.image_size[1] / 4, self.gf_dim * 2]
        deconv3_3 = relu(deconv2d(relu(tf.add(deconv4_1, res_connect[2])),
                                  output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv3_3', reuse=reuse))
        deconv3_2 = relu(deconv2d(deconv3_3, output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv3_2', reuse=reuse))
        deconv3_1 = relu(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv3_1', reuse=reuse))

        shapel2 = [self.batch_size, self.image_size[0] / 2,
                   self.image_size[1] / 2, self.gf_dim * 2]
        shapeout2 = [self.batch_size, self.image_size[0] / 2,
                     self.image_size[1] / 2, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                  output_shape=shapel2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv2_2', reuse=reuse))
        deconv2_1 = relu(deconv2d(deconv2_2, output_shape=shapeout2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='decoder_vgg_deconv2_1', reuse=reuse))

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                  output_shape=shapel1, k_h=3, k_w=3, d_h=1, d_w=1,
                                  name='decoder_vgg_deconv1_2', reuse=reuse))
        xtp1 = tanh(deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='decoder_vgg_deconv1_1', reuse=reuse))
        return xtp1


    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis0_conv'))

        h1_0 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis1_0_conv'),"bn1_0"))

        h1_1 = lrelu(batch_norm(conv2d(h1_0, self.df_dim * 2, name='dis1_1_conv'),"bn1_1"))

        h2_0 = lrelu(batch_norm(conv2d(h1_1, self.df_dim * 4, name='dis2_0_conv'),"bn2_0"))

        h2_1 = lrelu(batch_norm(conv2d(h2_0, self.df_dim * 4, name='dis2_1_conv'), "bn2_1"))

        h3 = lrelu(batch_norm(conv2d(h2_1, self.df_dim * 8, name='dis_h3_conv'), "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_linear')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "NETWORK.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None
