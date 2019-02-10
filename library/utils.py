import os
import errno
import numpy as np
import tensorflow as tf

def create_path(path):
    """Create path if not exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def txt_to_string(text_file_path):
    """Read text file as string"""

    f = open(text_file_path, 'r')
    txt_string = ''
    while True:
        single_line = f.readline()
        if single_line == '':
            break
        txt_string += single_line
    f.close()

    return txt_string


def create_rnn_dataset(txt_array, max_chars):
    """
    This function would return a training matrix and a label set
    matrix: # rows represents number of sentence observations
            # cols represents max number of chars in each sentence
    """
    txt_array_label = txt_array[1:]
    nrows = int(np.floor(txt_array.shape[0] / max_chars))
    txt_array = txt_array[:nrows*max_chars]
    txt_array_label = txt_array_label[:nrows*max_chars]

    txt_array = txt_array.reshape([-1, max_chars])
    txt_array_label = txt_array_label.reshape([-1, max_chars])

    return txt_array, txt_array_label


def train_val_split(train_mtx, label_mtx, train_proportion=0.8,
                    random_state=666):
    np.random.seed(random_state)
    num_train_rows = np.round(train_mtx.shape[0] * train_proportion).astype(int)
    rows_selected = np.random.choice(train_mtx.shape[0],
                                     num_train_rows, replace=False)
    rows_not_selected = list(
        set(range(train_mtx.shape[0])) - set(rows_selected))

    return (train_mtx[rows_selected], train_mtx[rows_not_selected],
            label_mtx[rows_selected], label_mtx[rows_not_selected])


class RNNDataset():
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()


class BatchManager():

    def __init__(self, train_set, num_epochs, shuffle=True,
                 random_state=666):
        """
        train_set, val_set: RNNDataset instances
        """
        self.train_set = train_set
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.current_epoch = 0
        self.rows_in_batch = []

    def next_batch(self, batch_size):
        """
        Output next batch as (X, y), return None if ran over num_epochs
        """
        num_rows = self.train_set.X.shape[0]

        while len(self.rows_in_batch) < batch_size:
            self.current_epoch += 1
            row_nums = list(range(num_rows))
            if self.shuffle:
                np.random.seed(self.random_state)
                np.random.shuffle(row_nums)
            self.rows_in_batch += row_nums

        selected_X = self.train_set.X[self.rows_in_batch[:batch_size]]
        selected_y = self.train_set.y[self.rows_in_batch[:batch_size]]
        self.rows_in_batch = self.rows_in_batch[batch_size:]

        if self.current_epoch > self.num_epochs:
            return None
        return selected_X, selected_y

def rnn_predict(model_path, char_string, char_encoder):
    char_array = np.array(list(char_string))
    char_array_num = char_encoder.transform(char_array)
    char_array_num = char_array_num[np.newaxis, :]

    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+'char_generator.ckpt.meta')
        saver.restore(sess, model_path+'char_generator.ckpt')

        graph = tf.get_default_graph()
        with graph.as_default():
            txt_input = graph.get_tensor_by_name('text_input:0')
            txt_input_next = graph.get_tensor_by_name('text_label:0')
            softmax_out = graph.get_tensor_by_name('output/Softmax:0')

            preds = sess.run(softmax_out, feed_dict={
                txt_input:char_array_num
            })

    return preds