"""
Lyrics generator
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from library.utils import rnn_predict

tf.logging.set_verbosity(0)

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--char_encoder_path',
        type=str,
        required=True,
        help='''Path to the character encoder generated from data_preprocessing.py'''
    )
    parser.add_argument(
        '--max_num_chars',
        type=int,
        required=True,
        default=100,
        help='''Max number of characters to generate'''
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='''Path to the trained tensorflow model'''
    )
    parser.add_argument(
        '--lyrics_start',
        type=str,
        required=True,
        help='''Charasters to start the lyrics'''
    )
    args = parser.parse_args()
    
    with open(args.char_encoder_path, 'rb') as f:
        char_encoder = pickle.load(f)

    lyrics = args.lyrics_start
    pred_char = list(lyrics)[-1]
    sentence_length = 200
    char_counter = 1
    space_counter = 0

    while char_counter <= sentence_length:
        preds = rnn_predict(args.model_path, lyrics, char_encoder)
        char_preds_df = pd.DataFrame([{'char':char, 'pred':pred} for char, pred in zip(char_encoder.classes_, preds[-1])])
        char_preds_df = char_preds_df.sort_values('pred', ascending=False)
        if (pred_char == '\n'):
            pred_char = np.random.choice(char_preds_df['char'], p=char_preds_df['pred']/char_preds_df['pred'].sum())
        else:
            pred_char = char_preds_df['char'].iloc[0]
        lyrics += pred_char
        print(lyrics)
        char_counter += 1
