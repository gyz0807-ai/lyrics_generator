"""
Preprocess text data
"""

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from library.utils import txt_to_string
from library.utils import train_val_split
from library.utils import create_rnn_dataset

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_file_path',
        type=str,
        required=True,
        help='''Path to the dataset'''
    )
    parser.add_argument(
        '--max_num_chars',
        type=int,
        required=True,
        default=40,
        help='''Max number of characters per observation'''
    )
    parser.add_argument(
        '--train_proportion',
        type=float,
        required=True,
        default=0.8,
        help='''Proportion of data to be used as training set'''
    )
    args = parser.parse_args()

    # read text file as string
    txt_string = txt_to_string(args.data_file_path)

    # convert characters to numbers
    txt_char_ls = list(txt_string)
    unique_chars = np.unique(txt_char_ls)
    le_char = LabelEncoder()
    le_char.fit(unique_chars)
    txt_num_ls = le_char.transform(txt_char_ls)
    txt_num_array = np.array(txt_num_ls)

    with open('./data/char_encoder.pkl', 'wb') as f:
        pickle.dump(le_char, f)

    # construct text dataset matrix and label set
    txt_mtx, txt_mtx_label = create_rnn_dataset(txt_num_array, args.max_num_chars)

    # create train test split
    train_x, val_x, train_y, val_y = train_val_split(
        txt_mtx, txt_mtx_label, train_proportion=args.train_proportion, random_state=666)

    with open('./data/train_x.pkl', 'wb') as f:
        pickle.dump(train_x, f)
    with open('./data/train_y.pkl', 'wb') as f:
        pickle.dump(train_y, f)
    with open('./data/val_x.pkl', 'wb') as f:
        pickle.dump(val_x, f)
    with open('./data/val_y.pkl', 'wb') as f:
        pickle.dump(val_y, f)
