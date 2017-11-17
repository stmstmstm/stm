import numpy as np 
import tensorflow as tf
import tensorlayer as tl
import os
import sys
import shutil
from tf_util import *
import cPickle as pickle
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log1]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 1000]')
parser.add_argument('--batch_size', type=int, default=100, help='Batch Size during training [default: 100]')
parser.add_argument('--learning_rate', type=float, default=1.0, help='Initial learning rate [default: 1.0]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 20000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--manifold_weight', type=float, default=0.2, help='Decay weight for manifold loss [default: 0.0]')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout pass rate [default: 0.6]')
parser.add_argument('--num_sample', type=int, default=30, help='Number of sample saved in the buffer [default: 30]')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay for conv layers [default: 0.00005]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
rec_loss_weight = FLAGS.manifold_weight
dp_pass_rate = FLAGS.dropout_rate
num_sample = FLAGS.num_sample
weight_decy_conv = FLAGS.weight_decay
write_result = True
padding_pixel = 4
lap_weight = 1

name_file = sys.argv[0]

if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.system('cp %s %s' % (name_file, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,
                        batch, 
                        DECAY_STEP,
                        DECAY_RATE,
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.0001)
    return learning_rate


def wide_block(input, stride, num_channel, name, is_training):
    with tf.variable_scope(name) as scope:
        w = batch_norm_for_conv2d(input, is_training, bn_decay=None, scope='bn1')
        w = tf.nn.relu(w)
        x = conv2d(w, num_channel, kernel_size=[3,3] , scope='conv_1', stride=[stride, stride], activation_fn=tf.nn.relu,
            bn=True, bn_decay=None, weight_decay=weight_decy_conv, is_training=is_training)
        x = dropout(x, is_training, 'drop_1', keep_prob=dp_pass_rate)
        x = conv2d(x, num_channel, kernel_size=[3,3], scope='conv_2', stride=[1,1], activation_fn=None, 
            bn=False, weight_decay=weight_decy_conv, is_training=is_training)
        y = conv2d(w, num_channel, kernel_size=[1,1], scope='conv_2_2', stride=[stride, stride], activation_fn=None, 
            bn=False, weight_decay=weight_decy_conv, is_training=is_training)
        y = tf.add(x,y,'adding_1')

        z = batch_norm_for_conv2d(y, is_training, bn_decay=None, scope='bn2')
        z = tf.nn.relu(z)
        z = conv2d(z, num_channel, kernel_size=[3,3], scope='conv3', stride=[1,1], activation_fn=tf.nn.relu, 
            bn=True, bn_decay=None, weight_decay=weight_decy_conv, is_training=is_training)
        z = dropout(z, is_training, 'drop_2', keep_prob=dp_pass_rate)
        z = conv2d(z, num_channel, kernel_size=[3,3], scope='conv4', stride=[1,1], activation_fn=None,
            bn=False, bn_decay=None, weight_decay=weight_decy_conv, is_training=is_training)

        z = tf.add(z, y, 'adding_2')

    return z

# def load_cifar_data():
#     a = np.load('./data/x_train.npy')
#     train_mean = a.mean()
#     train_std = a.std()
#     a = (a - train_mean) / train_std
#     b = np.load('./data/y_train.npy')
#     c = np.load('./data/x_test.npy')
#     c = (c - train_mean) / train_std
#     d = np.load('./data/y_test.npy')
#     return a,b,c,d

def load_cifar_data():
    x_train, y_train, x_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    x_train /= 255.0
    x_test /= 255.0
    train_mean = x_train.mean()
    train_std = x_train.std()
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std
    return x_train, y_train, x_test, y_test

def get_model(input_data, is_training):
    x = conv2d(input_data, num_output_channels=16, kernel_size=[3,3], scope='conv_init', stride=[1,1], activation_fn=None,
        bn=False, bn_decay=None, weight_decay=weight_decy_conv, is_training=is_training)
    x = wide_block(x, 1, 160, 'conv_block_1', is_training)
    x = wide_block(x, 2, 160, 'conv_block_2', is_training)
    x = wide_block(x, 2, 320, 'conv_block_3', is_training)
    x = wide_block(x, 2, 640, 'conv_block_4', is_training)
    x = batch_norm_for_conv2d(x, is_training, bn_decay=None, scope='bn_end')
    x = tf.nn.relu(x)
    x = avg_pool2d(x, kernel_size=[4,4], scope='ave_pool', stride=[1,1])
    x = tf.reshape(x, (BATCH_SIZE, 640))
    k = fully_connected(x, num_outputs=10, scope='fc', activation_fn=None, is_training=is_training)
    return k, x

def train_data(input_data, is_training):
    a = tf.pad(input_data, [[0,0],[padding_pixel,padding_pixel],[padding_pixel,padding_pixel],[0,0]])
    a = tf.random_crop(a, size=[BATCH_SIZE, 32, 32, 3])
    return tf.cond(is_training,
          lambda: a,
          lambda: input_data)

def cal_loss(output, label):
    with tf.name_scope('loss'):
        soft_out = tf.nn.softmax(output)
        label_oh = tf.one_hot(label, 10, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(-label_oh * tf.log(soft_out + 0.00000001))
        tf.summary.scalar('loss_ce', loss)
        tf.add_to_collection('losses', loss)
        loss_all = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss_all

def cal_rec_loss(data_in, label_in, feature_in, buffer_data, buffer_feature):
	for i in range(BATCH_SIZE):
		label_this = label_in[i]
		weight = tf.reshape(cal_Lap_dis(data_in[i], buffer_data[label_this]), shape=(num_sample,1))
		dis_feature = tf.reshape(tf.reduce_mean(tf.square(buffer_feature[label_this]-feature_in[i]), axis=[1]),shape=(num_sample,1))
		rec_loss = rec_loss_weight * tf.reduce_mean(dis_feature * weight)
		tf.add_to_collection('rec_loss', rec_loss)
	loss_rec = tf.add_n(tf.get_collection('rec_loss'), name='rec_loss')
	tf.add_to_collection('losses', loss_rec)
	tf.summary.scalar('loss_rec', loss_rec)
	return loss_rec

def cal_Lap_dis(data_a, data_b):
	l1_dis = tf.reduce_mean(tf.abs(data_a - data_b), axis=[1,2,3])
	Lap_dis = tf.exp(lap_weight * l1_dis)
	weight = 1.0/Lap_dis
	weight = weight/tf.reduce_sum(weight)
	return weight

def shuffle_data(data, labels):

    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def update_buffer(gt_data, raw_data, buffer_data, feature, buffer_feature):
	for i in range(BATCH_SIZE):
		buffer_feature[gt_data[i], 0:-1] = buffer_feature[gt_data[i], 1:]
		buffer_feature[gt_data[i], -1] = feature[i]
		buffer_data[gt_data[i], 0:-1] = buffer_data[gt_data[i], 1:]
		buffer_data[gt_data[i], -1] = raw_data[i]
	return buffer_data, buffer_feature

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            data_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 32, 32, 3))
            gt_ph = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
            buffer_data_ph = tf.placeholder(tf.float32, shape=(10, num_sample, 32, 32, 3))
            buffer_feature_ph = tf.placeholder(tf.float32, shape=(10, num_sample, 640))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            aug_data = train_data(data_ph, is_training_pl)
            xx, feature_last= get_model(aug_data, is_training_pl)
            rec_loss = cal_rec_loss(data_ph, gt_ph, feature_last, buffer_data_ph, buffer_feature_ph)

            loss_all = cal_loss(xx, gt_ph)
            tf.summary.scalar('loss_all', loss_all)
            batch = tf.Variable(0, trainable=False)
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            #optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            train_op = optimizer.minimize(loss_all, global_step=batch)
            correct_prediction = tf.equal(tf.argmax(xx,1), gt_ph)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        x_train, y_train, x_test, y_test = load_cifar_data()

        buffer_data = np.zeros((10, num_sample, 32, 32, 3))
       	buffer_feature = np.zeros((10, num_sample, 640))

        file_size = x_train.shape[0]
        num_batches = file_size/BATCH_SIZE

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        init = tf.global_variables_initializer()

        sess.run(init)
        count = 0
        for epoch_idx in range(MAX_EPOCH):
            current_data, current_label, _ = shuffle_data(x_train, y_train)
            for batch_idx in range(num_batches):
                count += 1
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                feed_data = current_data[start_idx:end_idx, ...]
                feed_label = current_label[start_idx:end_idx, ...]
                summary, step, _, current_loss, train_acc, feature_l = sess.run([merged, batch, 
                    train_op, loss_all, accuracy, feature_last], 
                    feed_dict={data_ph: feed_data,
                    gt_ph: feed_label,
                    buffer_feature_ph: buffer_feature,
                    buffer_data_ph: buffer_data,
                    is_training_pl: True})
                buffer_data, buffer_feature = update_buffer(
                	feed_label, feed_data, buffer_data, feature_l, buffer_feature)

                train_writer.add_summary(summary, step)
                if batch_idx % 50 == 0: 
                    log_string("Loss for Iter %d: %f, acc = %f" %(count,current_loss,train_acc))

                    #np.save('buffer_feature', buffer_feature)
                    #np.save('buffer_data', buffer_data)
            if epoch_idx % 1 == 0:
                test_iter = x_test.shape[0]/BATCH_SIZE
                current_data, current_label, _ = shuffle_data(x_test, y_test)
                loss_sum = 0
                acc_sum = 0
                re_list = []
                for test_idx in range(test_iter):
                    start_idx = test_idx * BATCH_SIZE
                    end_idx = (test_idx+1) * BATCH_SIZE
                    feed_data = current_data[start_idx:end_idx, ...]
                    feed_label = current_label[start_idx:end_idx, ...]
                    test_acc, result_t, loss_test= sess.run([accuracy, xx, loss_all], feed_dict={
                        data_ph: feed_data[0:BATCH_SIZE], 
                        gt_ph: feed_label[0: BATCH_SIZE], 
                        buffer_feature_ph: buffer_feature,
                        buffer_data_ph: buffer_data,
                        is_training_pl: False})
                    loss_sum += loss_test
                    acc_sum += test_acc
                    re_list.append(result_t)

                log_string("-----------------------------------" )
                log_string("Test loss for epoch %d: %f, Acc = %f" %(epoch_idx, loss_sum/float(test_iter), acc_sum/float(test_iter)))

            if epoch_idx % 10 == 0:
                if write_result == True:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
