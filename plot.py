"""
Critic网络代表网络的方向，若输出只为一维，则四个actor朝同一个方向，这并不合适，
因为四个网络应该有各自的方向，最终综合成一个方向,所以Critic网络必须对多个用户同时进行评价
即Critic网络的输出与用户个数是一致的
6/5
将分配带宽的actor整合成一个网络

"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mynet import myNet
import utils1
from config import options
import lib
import copy
import utils2

import scipy.io as sio

np.random.seed(0)

Debug = False
LOG_DIR = './log'
# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS=1

MAX_GLOBAL_EP = 1000  # the max iterations of global net
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # the global net update every UPDATE_GLOBAL_ITER steps
GAMMA = 0.9
ENTROPY_BETA = 0.05

LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic

GLOBAL_RUNNING_R = []  # total reward
GLOBAL_EP = 0


ENV_DIMS_new = 8
ACTION_DIMS = 40

SNR_data_length = 10


class A3Cnet(object):
    def __init__(self, scope, globalAC=None):
        self.s_CR = tf.placeholder(tf.float32, [None, 4 * ENV_DIMS_new], "allEnv")

        if scope == GLOBAL_NET_SCOPE:  # create global network
            with tf.variable_scope(scope):
                self.s1_CR = tf.placeholder(tf.float32, [None, ENV_DIMS_new], "Env1")
                self.s2_CR = tf.placeholder(tf.float32, [None, ENV_DIMS_new], "Env2")
                self.s3_CR = tf.placeholder(tf.float32, [None, ENV_DIMS_new], "Env3")
                self.s4_CR = tf.placeholder(tf.float32, [None, ENV_DIMS_new], "Env4")

                # ************************ Net Model ***********************************
                # Actor-Net
                self.CR1_A_prob, self.CR1_A_params = self._build_CR_Actor1(scope)
                self.CR2_A_prob, self.CR2_A_params = self._build_CR_Actor2(scope)
                self.CR3_A_prob, self.CR3_A_params = self._build_CR_Actor3(scope)
                self.CR4_A_prob, self.CR4_A_params = self._build_CR_Actor4(scope)


                # Critic-Net
                self.CR_v, self.CR_C_params = self._build_CR_Critic(scope)


        else:  # create worker network
            with tf.variable_scope(scope):
                self.s1_CR = tf.placeholder(tf.float32, [None,  ENV_DIMS_new], "Env1")
                self.s2_CR = tf.placeholder(tf.float32, [None,  ENV_DIMS_new], "Env2")
                self.s3_CR = tf.placeholder(tf.float32, [None,  ENV_DIMS_new], "Env3")
                self.s4_CR = tf.placeholder(tf.float32, [None,  ENV_DIMS_new], "Env4")

                self.cr1_a = tf.placeholder(tf.int32, [None, ACTION_DIMS], "action_cr1")
                self.cr2_a = tf.placeholder(tf.int32, [None, ACTION_DIMS], "action_cr2")
                self.cr3_a = tf.placeholder(tf.int32, [None, ACTION_DIMS], "action_cr3")
                self.cr4_a = tf.placeholder(tf.int32, [None, ACTION_DIMS], "action_cr4")

                self.CR_v_target = tf.placeholder(tf.float32, [None, 4], 'CR_Vtarget')

                # ********************************* Net Model ***************************************
                # CR Actor-Net
                self.CR1_prob, self.CR1_A_params = self._build_CR_Actor1(scope)
                self.CR2_prob, self.CR2_A_params = self._build_CR_Actor2(scope)
                self.CR3_prob, self.CR3_A_params = self._build_CR_Actor3(scope)
                self.CR4_prob, self.CR4_A_params = self._build_CR_Actor4(scope)

                self.c1_CR, c1_CRList_d = self.choose_CR_p(self.CR1_prob)
                self.c2_CR, c2_CRList_d = self.choose_CR_p(self.CR2_prob)
                self.c3_CR, c3_CRList_d = self.choose_CR_p(self.CR3_prob)
                self.c4_CR, c4_CRList_d = self.choose_CR_p(self.CR4_prob)

                self.capa2_all = options.serverCC - lib.CR_mapping[c1_CRList_d][0] * options.serverCC
                self.capa3_all = self.capa2_all - lib.CR_mapping[c2_CRList_d][0] * self.capa2_all
                self.capa4_all = self.capa3_all - lib.CR_mapping[c3_CRList_d][0] * self.capa3_all

                self.capa1_prob = lib.CR_mapping[c1_CRList_d][0]
                self.capa2_prob = (lib.CR_mapping[c2_CRList_d][0] * self.capa2_all) / options.serverCC
                self.capa3_prob = (lib.CR_mapping[c3_CRList_d][0] * self.capa3_all) / options.serverCC
                self.capa4_prob = (lib.CR_mapping[c4_CRList_d][0] * self.capa4_all) / options.serverCC


                # CR Critic-Net
                self.CR_v, self.CR_C_params = self._build_CR_Critic(scope)

                # ************************* Loss CR1 *********************************************
                self.CR1_v_target = self.CR_v_target[:, 0]
                self.CR1_v = self.CR_v[:, 0]
                self.CR1_td = tf.subtract(self.CR1_v_target, self.CR1_v, name='CR1_TD_error')
                with tf.name_scope('CR_C_loss'):
                    self.CR_C_loss = tf.reduce_sum(tf.square(self.CR1_td))

                with tf.name_scope('CR_A_loss'):
                    self.CR1_log = tf.log(self.CR1_prob + 1e-2)
                    self.CR1_hot = tf.one_hot(self.cr1_a, 40, dtype=tf.float32)
                    self.CR1_log_hot = tf.log(self.CR1_prob + 1e-2) * tf.one_hot(self.cr1_a, 40, dtype=tf.float32)
                    self.CR1_log_prob = tf.reduce_sum(
                        tf.log(self.CR1_prob + 1e-2) * tf.one_hot(self.cr1_a, 40, dtype=tf.float32), axis=1)

                    self.CR1_exp_v = self.CR1_log_prob * tf.stop_gradient(self.CR1_td)
                    self.CR1_entropy = - tf.reduce_sum(self.CR1_prob * tf.log(self.CR1_prob + 1e-2), axis=1)
                    self.CR1_exp_v = ENTROPY_BETA * self.CR1_entropy - self.CR1_exp_v
                    self.CR1_A_loss = tf.reduce_sum(self.CR1_exp_v)

                with tf.name_scope('CR_local_grad'):
                    self.CR1_A_grads = tf.gradients(self.CR1_A_loss, self.CR1_A_params)
                    self.CR_C_grads = tf.gradients(self.CR_C_loss, self.CR_C_params)

                # ************************* Loss CR2*********************************************
                self.CR2_v_target = self.CR_v_target[:, 1]
                self.CR2_v = self.CR_v[:, 1]
                self.CR2_td = tf.subtract(self.CR2_v_target, self.CR2_v, name='CR2_TD_error')
                with tf.name_scope('CR_C_loss'):
                    self.CR_C_loss = tf.reduce_sum(tf.square(self.CR2_td))

                with tf.name_scope('CR_A_loss'):
                    self.CR2_log = tf.log(self.CR2_prob + 1e-2)
                    self.CR2_hot = tf.one_hot(self.cr2_a, 40, dtype=tf.float32)
                    self.CR2_log_hot = tf.log(self.CR2_prob + 1e-2) * tf.one_hot(self.cr2_a, 40, dtype=tf.float32)
                    self.CR2_log_prob = tf.reduce_sum(
                        tf.log(self.CR2_prob + 1e-2) * tf.one_hot(self.cr2_a, 40, dtype=tf.float32), axis=1)

                    self.CR2_exp_v = self.CR2_log_prob * tf.stop_gradient(self.CR2_td)
                    self.CR2_entropy = - tf.reduce_sum(self.CR2_prob * tf.log(self.CR2_prob + 1e-2), axis=1)
                    self.CR2_exp_v = ENTROPY_BETA * self.CR2_entropy - self.CR2_exp_v
                    self.CR2_A_loss = tf.reduce_sum(self.CR2_exp_v)

                with tf.name_scope('CR_local_grad'):
                    self.CR2_A_grads = tf.gradients(self.CR2_A_loss, self.CR2_A_params)
                    self.CR_C_grads = tf.gradients(self.CR_C_loss, self.CR_C_params)

                # ************************* Loss CR3*********************************************
                self.CR3_v_target = self.CR_v_target[:, 2]
                self.CR3_v = self.CR_v[:, 2]
                self.CR3_td = tf.subtract(self.CR3_v_target, self.CR3_v, name='CR3_TD_error')
                with tf.name_scope('CR_C_loss'):
                    self.CR_C_loss = tf.reduce_sum(tf.square(self.CR3_td))

                with tf.name_scope('CR_A_loss'):
                    self.CR3_log = tf.log(self.CR3_prob + 1e-2)
                    self.CR3_hot = tf.one_hot(self.cr3_a, 40, dtype=tf.float32)
                    self.CR3_log_hot = tf.log(self.CR3_prob + 1e-2) * tf.one_hot(self.cr3_a, 40, dtype=tf.float32)
                    self.CR3_log_prob = tf.reduce_sum(
                        tf.log(self.CR3_prob + 1e-2) * tf.one_hot(self.cr3_a, 40, dtype=tf.float32), axis=1)

                    self.CR3_exp_v = self.CR3_log_prob * tf.stop_gradient(self.CR3_td)
                    self.CR3_entropy = - tf.reduce_sum(self.CR3_prob * tf.log(self.CR3_prob + 1e-2), axis=1)
                    self.CR3_exp_v = ENTROPY_BETA * self.CR3_entropy - self.CR3_exp_v
                    self.CR3_A_loss = tf.reduce_sum(self.CR3_exp_v)

                with tf.name_scope('CR_local_grad'):
                    self.CR3_A_grads = tf.gradients(self.CR3_A_loss, self.CR3_A_params)
                    self.CR_C_grads = tf.gradients(self.CR_C_loss, self.CR_C_params)

                # ************************* Loss CR4*********************************************
                self.CR4_v_target = self.CR_v_target[:, 3]
                self.CR4_v = self.CR_v[:, 3]
                self.CR4_td = tf.subtract(self.CR4_v_target, self.CR4_v, name='CR4_TD_error')
                with tf.name_scope('CR_C_loss'):
                    self.CR_C_loss = tf.reduce_sum(tf.square(self.CR4_td))

                with tf.name_scope('CR_A_loss'):
                    self.CR4_log = tf.log(self.CR4_prob + 1e-2)
                    self.CR4_hot = tf.one_hot(self.cr4_a, 40, dtype=tf.float32)
                    self.CR4_log_hot = tf.log(self.CR4_prob + 1e-2) * tf.one_hot(self.cr4_a, 40, dtype=tf.float32)
                    self.CR4_log_prob = tf.reduce_sum(
                        tf.log(self.CR4_prob + 1e-2) * tf.one_hot(self.cr4_a, 40, dtype=tf.float32), axis=1)

                    self.CR4_exp_v = self.CR4_log_prob * tf.stop_gradient(self.CR4_td)
                    self.CR4_entropy = - tf.reduce_sum(self.CR4_prob * tf.log(self.CR4_prob + 1e-2), axis=1)
                    self.CR4_exp_v = ENTROPY_BETA * self.CR4_entropy - self.CR4_exp_v
                    self.CR4_A_loss = tf.reduce_sum(self.CR4_exp_v)

                with tf.name_scope('CR_local_grad'):
                    self.CR4_A_grads = tf.gradients(self.CR4_A_loss, self.CR4_A_params)
                    self.CR_C_grads = tf.gradients(self.CR_C_loss, self.CR_C_params)

            with tf.name_scope('CR_sync'):  # global和worker的同步过程
                with tf.name_scope('CR_pull'):  # 把主网络的梯度参数赋予各子网络
                    self.pull_CR1_A_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                zip(self.CR1_A_params, globalAC.CR1_A_params)]
                    self.pull_CR2_A_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.CR2_A_params, globalAC.CR2_A_params)]
                    self.pull_CR3_A_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.CR3_A_params, globalAC.CR3_A_params)]
                    self.pull_CR4_A_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.CR4_A_params, globalAC.CR4_A_params)]
                    self.pull_CR_C_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                zip(self.CR_C_params, globalAC.CR_C_params)]

                with tf.name_scope('CR_push'):  # 使用子网络的梯度对主网络参数进行更新
                    self.update_CR1_A_params_op = OPT_CR_A.apply_gradients(zip(self.CR1_A_grads, globalAC.CR1_A_params))
                    self.update_CR2_A_params_op = OPT_CR_A.apply_gradients(zip(self.CR2_A_grads, globalAC.CR2_A_params))
                    self.update_CR3_A_params_op = OPT_CR_A.apply_gradients(zip(self.CR3_A_grads, globalAC.CR3_A_params))
                    self.update_CR4_A_params_op = OPT_CR_A.apply_gradients(zip(self.CR4_A_grads, globalAC.CR4_A_params))
                    self.update_CR_C_params_op = OPT_CR_C.apply_gradients(zip(self.CR_C_grads, globalAC.CR_C_params))


    def _build_CR_Actor1(self, scope):
        with tf.variable_scope('CR1'):
            self.CR_l1 = tf.layers.dense(
                inputs=self.s1_CR,
                units=512,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l1'
            )
            self.CR_l1_bn = tf.layers.batch_normalization(inputs=self.CR_l1, training=True)

            self.CR_l2 = tf.layers.dense(
                inputs=self.CR_l1_bn,
                units=128,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l2'
            )
            self.CR_l2_bn = tf.layers.batch_normalization(inputs=self.CR_l2, training=True)

            self.CR_l3 = tf.layers.dense(
                inputs=self.CR_l2_bn,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l3'
            )
            self.CR_l3_bn = tf.layers.batch_normalization(inputs=self.CR_l3, training=True)

            self.CR_l4 = tf.layers.dense(
                inputs=self.CR_l3_bn,
                units=40,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l4'
            )

            # todo:
            CR1_uni = tf.nn.softmax(self.CR_l4, axis=1)
            # print("CR1_uni", np.shape(CR1_uni))   # (none,40)
        CR1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/CR1')

        return CR1_uni, CR1_params

    def _build_CR_Actor2(self, scope):
        with tf.variable_scope('CR2'):
            self.CR_l1 = tf.layers.dense(
                inputs=self.s2_CR,
                units=512,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l1'
            )
            self.CR_l1_bn = tf.layers.batch_normalization(inputs=self.CR_l1, training=True)

            self.CR_l2 = tf.layers.dense(
                inputs=self.CR_l1_bn,
                units=128,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l2'
            )
            self.CR_l2_bn = tf.layers.batch_normalization(inputs=self.CR_l2, training=True)

            self.CR_l3 = tf.layers.dense(
                inputs=self.CR_l2_bn,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l3'
            )
            self.CR_l3_bn = tf.layers.batch_normalization(inputs=self.CR_l3, training=True)

            self.CR_l4 = tf.layers.dense(
                inputs=self.CR_l3_bn,
                units=40,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l4'
            )

            # todo:
            CR2_uni = tf.nn.softmax(self.CR_l4, axis=1)

        CR2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/CR2')

        return CR2_uni, CR2_params

    def _build_CR_Actor3(self, scope):
        with tf.variable_scope('CR3'):
            self.CR_l1 = tf.layers.dense(
                inputs=self.s3_CR,
                units=512,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l1'
            )
            self.CR_l1_bn = tf.layers.batch_normalization(inputs=self.CR_l1, training=True)

            self.CR_l2 = tf.layers.dense(
                inputs=self.CR_l1_bn,
                units=128,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l2'
            )
            self.CR_l2_bn = tf.layers.batch_normalization(inputs=self.CR_l2, training=True)

            self.CR_l3 = tf.layers.dense(
                inputs=self.CR_l2_bn,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l3'
            )
            self.CR_l3_bn = tf.layers.batch_normalization(inputs=self.CR_l3, training=True)

            self.CR_l4 = tf.layers.dense(
                inputs=self.CR_l3_bn,
                units=40,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l4'
            )

            # todo:
            CR3_uni = tf.nn.softmax(self.CR_l4, axis=1)

        CR3_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/CR3')

        return CR3_uni, CR3_params

    def _build_CR_Actor4(self, scope):
        with tf.variable_scope('CR4'):
            self.CR_l1 = tf.layers.dense(
                inputs=self.s4_CR,
                units=512,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l1'
            )
            self.CR_l1_bn = tf.layers.batch_normalization(inputs=self.CR_l1, training=True)

            self.CR_l2 = tf.layers.dense(
                inputs=self.CR_l1_bn,
                units=128,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l2'
            )
            self.CR_l2_bn = tf.layers.batch_normalization(inputs=self.CR_l2, training=True)

            self.CR_l3 = tf.layers.dense(
                inputs=self.CR_l2_bn,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l3'
            )
            self.CR_l3_bn = tf.layers.batch_normalization(inputs=self.CR_l3, training=True)

            self.CR_l4 = tf.layers.dense(
                inputs=self.CR_l3_bn,
                units=40,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='CR_l4'
            )

            CR4_uni = tf.nn.softmax(self.CR_l4, axis=1)

        CR4_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/CR4')

        return CR4_uni, CR4_params

    def _build_CR_Critic(self, scope):
        with tf.variable_scope('Critic_CR'):
            inputs = tf.layers.batch_normalization(self.s_CR, training=True)
            l1 = tf.layers.dense(
                inputs=inputs,
                units=512,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=128,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l2'
            )

            l3 = tf.layers.dense(
                inputs=l2,
                units=16,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l3'
            )

            v = tf.layers.dense(
                inputs=l3,
                units=4,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Critic_CR')
        return v, c_params  # v(none,4)

    def train_CR1(self, feed_dict):
        SESS.run([self.update_CR1_A_params_op, self.update_CR_C_params_op], feed_dict)
        SESS.run([self.pull_CR1_A_params_op, self.pull_CR_C_params_op], feed_dict)

    def train_CR2(self, feed_dict):
        SESS.run([self.update_CR2_A_params_op, self.update_CR_C_params_op], feed_dict)
        SESS.run([self.pull_CR2_A_params_op, self.pull_CR_C_params_op], feed_dict)

    def train_CR3(self, feed_dict):
        SESS.run([self.update_CR3_A_params_op, self.update_CR_C_params_op], feed_dict)
        SESS.run([self.pull_CR3_A_params_op, self.pull_CR_C_params_op], feed_dict)

    def train_CR4(self, feed_dict):
        SESS.run([self.update_CR4_A_params_op, self.update_CR_C_params_op], feed_dict)
        SESS.run([self.pull_CR4_A_params_op, self.pull_CR_C_params_op], feed_dict)

    def choose_CR_p(self, actor_prob_op):
        CRList_d = np.random.choice(lib.CR_mapping_index, 1, p=actor_prob_op[0])
        CR = lib.CR_mapping[CRList_d[0]]
        return CR, CRList_d[0]



class Worker(object):
    def __init__(self, name, globalAC, isPrint=False):
        self.net = myNet()
        self.net.createNetTopo()
        self.isPrint = isPrint
        self.name = name  # name of worker
        self.AC = A3Cnet(name, globalAC)


    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP  # GLOBAL_RUNNING_R is the reward of all workers, GLOBAL_EP is the total iterations of all workers
        total_step = 1  # iterations of this worker

        # Store the train_data and train_label
        allSNR = [[] for h_index in range(options.HostNum)]
        # Start to simulate the video-downloading
        self.clientsExecResult = self.net.updateClientVideo()
        allClientSNR = utils2.unitEnv_uni(self.clientsExecResult)
        for h_index in range(options.HostNum):
            allSNR[h_index] += allClientSNR[h_index].tolist()

        buffer_s, buffer1_s,buffer2_s, buffer3_s, buffer4_s,\
        buffer_CR_a, buffer1_CR_a, buffer2_CR_a, buffer3_CR_a, buffer4_CR_a,\
        buffer_CR1_r, buffer_CR2_r, buffer_CR3_r, buffer_CR4_r = [], [], [], [], [], [], [], [], [], [], [], [] ,[], []
        windowInfo = []
        rewardCRList = [[] for _ in range(options.HostNum)]

        # while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
        while GLOBAL_EP < MAX_GLOBAL_EP:
            # ep_r = 0  # the total reward of this episode

            while True:
                allClientsAction = {}
                c1_action = {}
                c2_action = {}
                c3_action = {}
                c4_action = {}


                # get the env info
                env, *s_env = utils1.env_state8(self.clientsExecResult)

                feed_dict1 = {self.AC.s1_CR: np.array(env[0]).reshape((-1, ENV_DIMS_new))}
                feed_dict2 = {self.AC.s2_CR: np.array(env[1]).reshape((-1, ENV_DIMS_new))}
                feed_dict3 = {self.AC.s3_CR: np.array(env[2]).reshape((-1, ENV_DIMS_new))}
                feed_dict4 = {self.AC.s4_CR: np.array(env[3]).reshape((-1, ENV_DIMS_new))}

                CR1_prob = SESS.run(self.AC.CR1_prob, feed_dict1)
                CR2_prob = SESS.run(self.AC.CR2_prob, feed_dict2)
                CR3_prob = SESS.run(self.AC.CR3_prob, feed_dict3)
                CR4_prob = SESS.run(self.AC.CR4_prob, feed_dict4)


                c1_CRList, c1_CRList_d = self.AC.choose_CR_p(CR1_prob)
                c2_CRList, c2_CRList_d = self.AC.choose_CR_p(CR2_prob)
                c3_CRList, c3_CRList_d = self.AC.choose_CR_p(CR3_prob)
                c4_CRList, c4_CRList_d = self.AC.choose_CR_p(CR4_prob)

                capa2_all = options.serverCC - lib.CR_mapping[c1_CRList_d][0] * options.serverCC
                capa3_all = capa2_all - lib.CR_mapping[c2_CRList_d][0] * capa2_all
                capa4_all = capa3_all - lib.CR_mapping[c3_CRList_d][0] * capa3_all

                # add buffer info
                capa1_prob = lib.CR_mapping[c1_CRList_d][0]
                env[0][-1] = capa1_prob

                capa2_prob = (lib.CR_mapping[c2_CRList_d][0] * capa2_all) / options.serverCC
                env[1][-1] = capa2_prob

                capa3_prob = (lib.CR_mapping[c3_CRList_d][0] * capa3_all) / options.serverCC
                env[2][-1] = capa3_prob

                capa4_prob = (lib.CR_mapping[c4_CRList_d][0] * capa4_all) / options.serverCC
                env[3][-1] = capa4_prob

                allenv = np.concatenate([env[0], env[1], env[2], env[3]], axis=0)
                buffer_s.append(np.array(allenv))
                buffer1_s.append(np.array(env[0]))
                buffer2_s.append(np.array(env[1]))
                buffer3_s.append(np.array(env[2]))
                buffer4_s.append(np.array(env[3]))


                CR_prob = np.concatenate([CR1_prob, CR2_prob, CR3_prob, CR4_prob], 0)
                buffer1_CR_a.append(CR1_prob)
                buffer2_CR_a.append(CR2_prob)
                buffer3_CR_a.append(CR3_prob)
                buffer4_CR_a.append(CR4_prob)
                buffer_CR_a.append(CR_prob)  # todo
                print("CR_prob", CR_prob)
                print("CR_prob", np.shape(CR_prob))  # (4,40)
                print("buffer_CR_a", np.shape(buffer_CR_a))  # (9,4,40)


                c1_action["CC"] = lib.CR_mapping[c1_CRList_d][0] * options.serverCC
                c2_action["CC"] = lib.CR_mapping[c2_CRList_d][0] * capa2_all
                c3_action["CC"] = lib.CR_mapping[c3_CRList_d][0] * capa3_all
                c4_action["CC"] = lib.CR_mapping[c4_CRList_d][0] * capa4_all


                c1_action["RR"] = c1_CRList[1]
                c2_action["RR"] = c2_CRList[1]
                c3_action["RR"] = c3_CRList[1]
                c4_action["RR"] = c4_CRList[1]

                allClientsAction['c1'] = c1_action
                print("c1Action",c1_action)
                allClientsAction['c2'] = c2_action
                allClientsAction['c3'] = c3_action
                allClientsAction['c4'] = c4_action
                print("allAction:",allClientsAction)
                # update env_state according to the CC and resolution choices
                self.clientsExecResult = self.net.updateClientVideo(allClientsAction)

                # Use window to record the info
                windowInfo.append(copy.deepcopy(self.clientsExecResult))
                if len(windowInfo) > 5:
                    del windowInfo[0]

                # compute reward
                # r = utils.reward_window(windowInfo)
                ep_r_CR1,  ep_r_CR2, ep_r_CR3, ep_r_CR4 = utils1.reward_joint3(self.clientsExecResult)   # todo:reward_joint3
                buffer_CR1_r.append(ep_r_CR1)
                buffer_CR2_r.append(ep_r_CR2)
                buffer_CR3_r.append(ep_r_CR3)
                buffer_CR4_r.append(ep_r_CR4)

                rewardCRList[0].append(copy.deepcopy(ep_r_CR1))
                rewardCRList[1].append(copy.deepcopy(ep_r_CR2))
                rewardCRList[2].append(copy.deepcopy(ep_r_CR3))
                rewardCRList[3].append(copy.deepcopy(ep_r_CR4))


                capa2_all = options.serverCC - CR1_prob[0] * options.serverCC
                capa3_all = capa2_all - CR2_prob[0] * capa2_all
                capa4_all = capa3_all - CR3_prob[0] * capa3_all

                if total_step % 1 == 0:
                    print("CC_AAAA1: ", lib.CR_mapping[c1_CRList_d][0], lib.CR_mapping[c1_CRList_d][0] * options.serverCC)
                    print("CC_AAAA2: ", lib.CR_mapping[c2_CRList_d][0], lib.CR_mapping[c2_CRList_d][0] * capa2_all)
                    print("CC_AAAA3: ", lib.CR_mapping[c3_CRList_d][0], lib.CR_mapping[c3_CRList_d][0] * capa3_all)
                    print("CC_AAAA4: ", lib.CR_mapping[c4_CRList_d][0], lib.CR_mapping[c4_CRList_d][0] * capa4_all)
                    print("-" * 30)   #
                    print("Reso_AAAA1: ", c1_CRList[1])
                    print("Reso_AAAA2: ", c2_CRList[1])
                    print("Reso_AAAA3: ", c3_CRList[1])
                    print("Reso_AAAA4: ", c4_CRList[1])

                total_step += 1
                if total_step % UPDATE_GLOBAL_ITER < 0:  # update global and assign to local net
                    GLOBAL_EP += 1
                    break

                if total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    print("GLOBAL_EP:", GLOBAL_EP)
                    # if self.isPrint:
                    #     self.printMidInfo()

                    env,  *s_env = utils1.env_state8(self.clientsExecResult)

                    feed_dict = {self.AC.s_CR: np.array(env).reshape((-1, 4 * ENV_DIMS_new))}
                    CR_v_= SESS.run(self.AC.CR_v, feed_dict)
                    print("CR_v_: ", CR_v_)
                    print("buffer_CR1_r:", buffer_CR1_r)
                    print("buffer_CR2_r:", buffer_CR2_r)
                    print("buffer_CR3_r:", buffer_CR3_r)
                    print("buffer_CR4_r:", buffer_CR4_r)

                    CR_v_target = [[] for _ in range(options.HostNum)]

                    for h_index in range(options.HostNum):
                        reward_CR_client = rewardCRList[h_index][::-1]
                        value = CR_v_[0, h_index]
                        for r in reward_CR_client:
                            value = r + GAMMA * value
                            CR_v_target[h_index].append(value)
                        CR_v_target[h_index].reverse()
                    CR_v_target = np.array(CR_v_target).T


                    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    # batch_num = len(buffer_s)
                    # # allENV = np.array(buffer_s).reshape((batch_num, 4 * ENV_DIMS_new))
                    # # allCR = np.array(buffer_CR_a).reshape((batch_num, 4))
                    #
                    #
                    # ENV1 = np.array(buffer_s[:0:]).reshape((batch_num, ENV_DIMS_new))
                    # ENV2 = np.array(buffer_s[1]).reshape((batch_num, ENV_DIMS_new))
                    # ENV3 = np.array(buffer_s[2]).reshape((batch_num, ENV_DIMS_new))
                    # ENV4 = np.array(buffer_s[3]).reshape((batch_num, ENV_DIMS_new))

                    # *****************************************************************************************
                    ENV1 = buffer1_s
                    ENV2 = buffer2_s
                    ENV3 = buffer3_s
                    ENV4 = buffer4_s
                    ALLENV = buffer_s

                    allCR1 = np.array(buffer1_CR_a)
                    allCR2 = np.array(buffer2_CR_a)
                    allCR3 = np.array(buffer3_CR_a)
                    allCR4 = np.array(buffer4_CR_a)
                    # *****************************************************************************************
                    feed_dict_A1 = {
                        self.AC.s_CR: ALLENV,    # (?, 32)   (9,32)
                        self.AC.s1_CR: ENV1,      # (?, 8)  (9,8)
                        self.AC.cr1_a: allCR1,   # (?, 1, 40)  (9,1,40)
                        self.AC.CR_v_target: np.reshape(CR_v_target, (-1, 4))   # (?,4)   (4,9)
                    }

                    feed_dict_A2 = {
                        self.AC.s_CR: ALLENV,
                        self.AC.s2_CR: ENV2,
                        self.AC.cr2_a: allCR2,
                        self.AC.CR_v_target: np.reshape(CR_v_target, (-1, 4))
                    }

                    feed_dict_A3 = {
                        self.AC.s_CR: ALLENV,
                        self.AC.s3_CR: ENV3,
                        self.AC.cr2_a: allCR3,
                        self.AC.CR_v_target: np.reshape(CR_v_target, (-1, 4))
                    }

                    feed_dict_A4 = {
                        self.AC.s_CR: ALLENV,
                        self.AC.s4_CR: ENV4,
                        self.AC.cr2_a: allCR4,
                        self.AC.CR_v_target: np.reshape(CR_v_target, (-1, 4))
                    }

                    # *********************************** Debug ******************************************************
                    CR1_A_loss = SESS.run(self.AC.CR1_A_loss, feed_dict_A1)
                    print("*" * 30)
                    print("CR1_A_loss:", CR1_A_loss)

                    CR2_A_loss = SESS.run(self.AC.CR2_A_loss, feed_dict_A2)
                    print("*" * 30)
                    print("CR2_A_loss:", CR2_A_loss)

                    CR3_A_loss = SESS.run(self.AC.CR3_A_loss, feed_dict_A3)
                    print("*" * 30)
                    print("CR3_A_loss:", CR3_A_loss)

                    CR4_A_loss = SESS.run(self.AC.CR4_A_loss, feed_dict_A4)
                    print("*" * 30)
                    print("CR4_A_loss:", CR4_A_loss)


                    # ************************************ Train *****************************************************
                    time = 3   # todo
                    for _ in range(time):
                        self.AC.train_CR1(feed_dict_A1)
                        self.AC.train_CR2(feed_dict_A2)
                        self.AC.train_CR3(feed_dict_A3)
                        self.AC.train_CR4(feed_dict_A4)

                    rewardCRList = [[] for _ in range(options.HostNum)]
                    buffer_s, buffer1_s, buffer2_s, buffer3_s, buffer4_s, \
                    buffer_CR_a, buffer1_CR_a, buffer2_CR_a, buffer3_CR_a, buffer4_CR_a, \
                    buffer_CR1_r, buffer_CR2_r, buffer_CR3_r, buffer_CR4_r = [], [], [], [], [], [], [], [], [], [], [], [], [], []

                    GLOBAL_EP += 1
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r_CR1)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r_CR1)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_CR_r: %i" % GLOBAL_RUNNING_R[-1],
                    )

                    break


if __name__ == "__main__":
    DISCR = []
    SNR = []

    vlist = []
    vtlist = []
    error = []
    episode_r = []
    aloss = []

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_CR_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_CR_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

        GLOBAL_AC = A3Cnet(GLOBAL_NET_SCOPE)

        workers = []

        # for i in range(N_WORKERS-1):
        #     i_name = 'Worker_%i' % i
        #     if i == N_WORKERS - 2:
        #         workers.append(Worker(i_name, GLOBAL_AC, isPrint=True))
        #     else:
        #         workers.append(Worker(i_name, GLOBAL_AC, isPrint=False))

        worker = Worker('worker_%i' % (N_WORKERS - 1), GLOBAL_AC, isPrint=True)
        workers.append(worker)

    saver = tf.train.Saver(max_to_keep=1)

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    # saver.save(SESS, options.modelSaveDir)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R, color="black")
    # plt.plot(np.arange(len(GLOBAL_RUNNING_Reso_R)), GLOBAL_RUNNING_Reso_R, color="red")
    # plt.plot(np.arange(len(episode_r)), episode_r, color="blue")
    # plt.plot(np.arange(len(error)), error, color="red")
    # plt.plot(np.arange(len(vlist)), vlist, color="green")
    # plt.plot(np.arange(len(vtlist)), vtlist, color="yellow")
    # plt.plot(np.arange(len(aloss)), aloss, color="black")


    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

