"""
Run recurrent neural network for training the lyrics generator
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import flags
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from library.utils import RNNDataset
from library.utils import create_path
from library.utils import BatchManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = flags.FLAGS

with open('./data/char_encoder.pkl', 'rb') as f:
    char_encoder = pickle.load(f)

if __name__ == '__main__':
    flags.DEFINE_integer('embedding_size', 100,
                         'Number of units in embedding layer')
    flags.DEFINE_list('num_rnn_layer_units', [128, 128],
                      'Number of units in lstm cells')
    flags.DEFINE_float('keep_prob', 0.8,
                       'Probabily for lstm nodes to be kept')
    flags.DEFINE_float('learning_rate', 1e-4,
                       'Learning rate for Adam Optimizer')
    flags.DEFINE_integer('batch_size', 500,
                         'Batch size for training set')
    flags.DEFINE_integer('num_epochs', 200,
                         'Number of epochs')
    flags.DEFINE_boolean('shuffle', True,
                         'Whether shuffle the training set')
    flags.DEFINE_integer('eval_frequency', 10,
                         'Number of steps between validation set '
                         'evaluations or model file updates')
    flags.DEFINE_integer('early_stopping_eval_rounds', 5,
                         'Perform early stop if the loss does '
                         'not drop in x evaluation rounds')
    flags.DEFINE_integer('vocab_size', char_encoder.classes_.shape[0],
                         'Number of chars in vocabulary')

class CharGenerator():

    def __init__(self):
        self.nn_dict = {}

    def generate_graph(self):
        tf.reset_default_graph()
        self.nn_dict['graph'] = tf.Graph()

        time_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        tf_log_dir = './tf_log/run-{}/'.format(time_now)
        self.nn_dict['tf_model_dir'] = './tf_model/model-{}/'.format(time_now)

        with self.nn_dict['graph'].as_default():
            self.nn_dict['txt_input'] = tf.placeholder(tf.int32, [None, None], 'text_input')
            self.nn_dict['txt_input_next'] = tf.placeholder(tf.int32, [None, None], 'text_label')
            txt_input_next_onehot = tf.one_hot(self.nn_dict['txt_input_next'], depth=FLAGS.vocab_size,
                                            axis=2, dtype=tf.int32, name='text_label_onehot')

            with tf.variable_scope('embedding'):
                embed_matrix = tf.Variable(tf.random_uniform(
                    [FLAGS.vocab_size, FLAGS.embedding_size]))
                txt_embedded = tf.nn.embedding_lookup(embed_matrix, self.nn_dict['txt_input'])

            with tf.variable_scope('rnn_1'):
                lstm_1 = tf.nn.rnn_cell.BasicLSTMCell(int(FLAGS.num_rnn_layer_units[0]), activation=tf.nn.tanh)
                lstm_dropout_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=FLAGS.keep_prob)
                rnn_1 = tf.nn.dynamic_rnn(lstm_dropout_1, txt_embedded, dtype=tf.float32)

            with tf.variable_scope('rnn_2'):
                lstm_2 = tf.nn.rnn_cell.BasicLSTMCell(int(FLAGS.num_rnn_layer_units[1]), activation=tf.nn.tanh)
                lstm_dropout_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=FLAGS.keep_prob)
                rnn_2 = tf.nn.dynamic_rnn(lstm_dropout_2, rnn_1[0], dtype=tf.float32)

            # with tf.variable_scope('concat'):
            #     concat_out = tf.concat([txt_embedded, rnn_1[0], rnn_2[0]], axis=2)

            with tf.variable_scope('output'):
                logit_out = tf.layers.dense(rnn_2[0], FLAGS.vocab_size)
                self.nn_dict['softmax_out'] = tf.nn.softmax(logit_out)

            with tf.variable_scope('loss'):
                loss_word = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=txt_input_next_onehot, logits=logit_out)
                loss_sum_sentence = tf.reduce_sum(loss_word, axis=1)
                self.nn_dict['loss_avg_batch'] = tf.reduce_mean(loss_sum_sentence)

            with tf.variable_scope('optimization'):
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                self.nn_dict['train'] = optimizer.minimize(self.nn_dict['loss_avg_batch'])

        self.nn_dict['train_loss_summary'] = tf.summary.scalar(
            'Train_Loss', self.nn_dict['loss_avg_batch'])
        self.nn_dict['val_loss_summary'] = tf.summary.scalar(
            'Validation_Loss', self.nn_dict['loss_avg_batch'])
        self.nn_dict['file_writer'] = tf.summary.FileWriter(tf_log_dir, self.nn_dict['graph'])

    def run_graph(self):
        bst_score = 99999
        step_counter = 1
        early_stopping_counter = 0

        with open('./data/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
        with open('./data/train_y.pkl', 'rb') as f:
            train_y = pickle.load(f)
        with open('./data/val_x.pkl', 'rb') as f:
            val_x = pickle.load(f)
        with open('./data/val_y.pkl', 'rb') as f:
            val_y = pickle.load(f)
        train_set = RNNDataset(train_x, train_y)
        val_set = RNNDataset(val_x, val_y)
        batch_manager = BatchManager(train_set, num_epochs=FLAGS.num_epochs,
                                    shuffle=FLAGS.shuffle, random_state=666)

        with self.nn_dict['graph'].as_default():
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)

                while True:
                    batch = batch_manager.next_batch(FLAGS.batch_size)
                    if batch is None:
                        break
                    batch_x, batch_y = batch[0], batch[1]

                    if step_counter % FLAGS.eval_frequency == 0:
                        train_loss = sess.run(self.nn_dict['loss_avg_batch'], feed_dict={
                            self.nn_dict['txt_input']:batch_x,
                            self.nn_dict['txt_input_next']:batch_y
                        })

                        val_loss = sess.run(self.nn_dict['loss_avg_batch'], feed_dict={
                            self.nn_dict['txt_input']:val_set.X,
                            self.nn_dict['txt_input_next']:val_set.y
                        })

                        print('Training Loss: {} | Validation Loss: {}'.format(
                            train_loss, val_loss))

                        summary_train_loss = sess.run(self.nn_dict['train_loss_summary'], feed_dict={
                            self.nn_dict['txt_input']:batch_x,
                            self.nn_dict['txt_input_next']:batch_y
                        })
                        self.nn_dict['file_writer'].add_summary(summary_train_loss, step_counter)

                        summary_val_loss = sess.run(self.nn_dict['val_loss_summary'], feed_dict={
                            self.nn_dict['txt_input']:val_set.X,
                            self.nn_dict['txt_input_next']:val_set.y
                        })
                        self.nn_dict['file_writer'].add_summary(summary_val_loss, step_counter)

                        if val_loss < bst_score:
                            early_stopping_counter = 0
                            saver.save(sess, self.nn_dict['tf_model_dir']+'char_generator.ckpt')
                        else:
                            early_stopping_counter += 1

                        if early_stopping_counter > FLAGS.early_stopping_eval_rounds:
                            break

                    sess.run(self.nn_dict['train'], feed_dict={
                        self.nn_dict['txt_input']:batch_x,
                        self.nn_dict['txt_input_next']:batch_y
                    })

                    step_counter += 1

            sess.close()

def main(argv=None):
    char_generator = CharGenerator()
    char_generator.generate_graph()
    char_generator.run_graph()

if __name__ == '__main__':
    tf.app.run()
