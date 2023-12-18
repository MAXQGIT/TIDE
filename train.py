# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main training code."""

import json
import os
import random
import string
import sys
import argparse

from absl import logging
import data_loader
import models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tider Long Sequences Forecasting')
parser.add_argument('--train_epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')
parser.add_argument('--epoch_len', default=None, help='number of iterations in an epoch')
parser.add_argument('--batch_size', default=512, help='Batch size for the randomly sampled batch')  # 512
parser.add_argument('--learning_rate', default=1e-4, help='Learning rate')
parser.add_argument('--expt_dir', default='./results', help='The name of the experiment dir', )
parser.add_argument('--dataset', default='weather', help='The name of the dataset.')
parser.add_argument('--datetime_col', default='date', help='Column having datetime.')
parser.add_argument('--num_cov_cols', default=None, help='Column having numerical features.')
parser.add_argument('--cat_cov_cols', default=None, help='Column having categorical features.')
parser.add_argument('--hist_len', default=720, help='Length of the history provided as input')  # 720
parser.add_argument('--pred_len', default=96, help='Length of pred len during training')
parser.add_argument('--num_layers', default=2, help='Number of DNN layers')
parser.add_argument('--hidden_size', default=256, help='Hidden size of DNN')  # 256
parser.add_argument('--decoder_output_dim', default=4, help='Hidden d3 of DNN')
parser.add_argument('--final_decoder_hidden', default=64, help='Hidden d3 of DNN')  # 64
parser.add_argument('--ts_cols', default=None, help='Columns of time-series features')
parser.add_argument('--random_seed', default=1, help='The random seed to be used for TF and numpy')
parser.add_argument('--normalize', default=True, help='normalize data for training or not')
parser.add_argument('--holiday', default=False, help='use holiday features or not')
parser.add_argument('--permute', default=True, help='permute the order of TS in training set')
parser.add_argument('--transform', default=False, help='Apply chronoml transform or not.')
parser.add_argument('--layer_norm', default=False, help='Apply layer norm or not.')
parser.add_argument('--dropout_rate', default=0.0, help='dropout rate')
parser.add_argument('--num_split', default=1, help='number of splits during inference.')
parser.add_argument('--min_num_epochs', default=0, help='minimum number of epochs before early stopping')
parser.add_argument('--gpu', default=0, help='index of gpu to be used.')
FLAGS = parser.parse_args()
DATA_DICT = {
    'ettm2': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETTm2.csv',
        'freq': '15min',
    },
    'ettm1': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETTm1.csv',
        'freq': '15min',
    },
    'etth2': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETTh2.csv',
        'freq': 'H',
    },
    'etth1': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETTh1.csv',
        'freq': 'H',
    },
    'elec': {
        'boundaries': [18413, 21044, 26304],
        'data_path': './datasets/electricity.csv',
        'freq': 'H',
    },
    'traffic': {
        'boundaries': [12280, 14036, 17544],
        'data_path': './datasets/traffic.csv',
        'freq': 'H',
    },
    'weather': {
        'boundaries': [36887, 42157, 52696],
        'data_path': './datasets/weather.csv',
        'freq': '10min',
    },
}

np.random.seed(1024)
tf.random.set_seed(1024)


def _get_random_string(num_chars):
    rand_str = ''.join(
        random.choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits
        )
        for _ in range(num_chars - 1)
    )
    return rand_str


def training():
    """Training TS code."""
    tf.random.set_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    experiment_id = _get_random_string(8)
    logging.info('Experiment id: %s', experiment_id)

    dataset = FLAGS.dataset
    data_path = DATA_DICT[dataset]['data_path']
    freq = DATA_DICT[dataset]['freq']
    boundaries = DATA_DICT[dataset]['boundaries']

    data_df = pd.read_csv(open(data_path, 'r', encoding='utf-8'))

    if FLAGS.ts_cols:
        ts_cols = DATA_DICT[dataset]['ts_cols']
        num_cov_cols = DATA_DICT[dataset]['num_cov_cols']
        cat_cov_cols = DATA_DICT[dataset]['cat_cov_cols']
    else:
        ts_cols = [col for col in data_df.columns if col != FLAGS.datetime_col]
        num_cov_cols = None
        cat_cov_cols = None
    permute = FLAGS.permute
    dtl = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col=FLAGS.datetime_col,
        num_cov_cols=num_cov_cols,
        cat_cov_cols=cat_cov_cols,
        ts_cols=np.array(ts_cols),
        train_range=[0, boundaries[0]],
        val_range=[boundaries[0], boundaries[1]],
        test_range=[boundaries[1], boundaries[2]],
        hist_len=FLAGS.hist_len,
        pred_len=FLAGS.pred_len,
        batch_size=min(FLAGS.batch_size, len(ts_cols)),
        freq=freq,
        normalize=FLAGS.normalize,
        epoch_len=FLAGS.epoch_len,
        holiday=FLAGS.holiday,
        permute=permute,
    )

    # Create model
    model_config = {
        'model_type': 'dnn',
        'hidden_dims': [FLAGS.hidden_size] * FLAGS.num_layers,
        'time_encoder_dims': [64, 4],
        'decoder_output_dim': FLAGS.decoder_output_dim,
        'final_decoder_hidden': FLAGS.final_decoder_hidden,
        'batch_size': dtl.batch_size,
    }
    model = models.TideModel(
        model_config=model_config,
        pred_len=FLAGS.pred_len,
        num_ts=len(ts_cols),
        cat_sizes=dtl.cat_sizes,
        transform=FLAGS.transform,
        layer_norm=FLAGS.layer_norm,
        dropout_rate=FLAGS.dropout_rate,
    )

    # Compute path to experiment directory
    expt_dir = os.path.join(
        FLAGS.expt_dir,
        FLAGS.dataset + '_' + str(experiment_id) + '_' + str(FLAGS.pred_len),
    )
    os.makedirs(expt_dir, exist_ok=True)

    step = tf.Variable(0)
    # LR scheduling
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FLAGS.learning_rate,
        decay_steps=30 * dtl.train_range[1],
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)
    summary = Summary(expt_dir)

    best_loss = np.inf
    pat = 0
    mean_loss_array = []
    iter_array = []
    # best_check_path = None
    while step.numpy() < FLAGS.train_epochs + 1:
        ep = step.numpy()
        logging.info('Epoch %s', ep)
        sys.stdout.flush()

        iterator = tqdm(dtl.tf_dataset(mode='train'), mininterval=2)
        for i, batch in enumerate(iterator):
            past_data = batch[:3]
            future_features = batch[4:6]
            tsidx = batch[-1]
            loss = model.train_step(
                past_data, future_features, batch[3], tsidx, optimizer
            )
            # Train metrics
            summary.update({'train/reg_loss': loss, 'train/loss': loss})
            if i % 100 == 0:
                mean_loss = summary.metric_dict['train/reg_loss'].result().numpy()
                mean_loss_array.append(mean_loss)
                iter_array.append(i)
                iterator.set_description(f'Loss {mean_loss:.4f}')
        step.assign_add(1)
        # Test metrics
        val_metrics, val_res, val_loss = model.eval(
            dtl, 'val', num_split=FLAGS.num_split
        )
        test_metrics, test_res, test_loss = model.eval(
            dtl, 'test', num_split=FLAGS.num_split
        )
        logging.info('Val Loss: %s', val_loss)
        logging.info('Test Loss: %s', test_loss)
        tracked_loss = val_metrics['rmse']
        if tracked_loss < best_loss and ep > FLAGS.min_num_epochs:
            best_loss = tracked_loss
            pat = 0

            with open(os.path.join(expt_dir, 'val_pred.npy'), 'wb') as fp:
                np.save(fp, val_res[0][:, 0: -1: FLAGS.pred_len])
            with open(os.path.join(expt_dir, 'val_true.npy'), 'wb') as fp:
                np.save(fp, val_res[1][:, 0: -1: FLAGS.pred_len])

            with open(os.path.join(expt_dir, 'test_pred.npy'), 'wb') as fp:
                np.save(fp, test_res[0][:, 0: -1: FLAGS.pred_len])
            with open(os.path.join(expt_dir, 'test_true.npy'), 'wb') as fp:
                np.save(fp, test_res[1][:, 0: -1: FLAGS.pred_len])
            with open(os.path.join(expt_dir, 'test_metrics.json'), 'w') as fp:
                json.dump(test_metrics, fp)
            logging.info('saved best result so far at %s', expt_dir)
            logging.info('Test metrics: %s', test_metrics)
        else:
            pat += 1
            if pat > FLAGS.patience:
                logging.info('Early stopping')
                break
        summary.write(step=step.numpy())


class Summary:
    """Summary statistics."""

    def __init__(self, log_dir):
        self.metric_dict = {}
        self.writer = tf.summary.create_file_writer(log_dir)

    def update(self, update_dict):
        for metric in update_dict:
            if metric not in self.metric_dict:
                self.metric_dict[metric] = keras.metrics.Mean()
            self.metric_dict[metric].update_state(values=[update_dict[metric]])

    def write(self, step):
        with self.writer.as_default():
            for metric in self.metric_dict:
                tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
        self.metric_dict = {}
        self.writer.flush()


def main():
    training()


if __name__ == '__main__':
    main()
