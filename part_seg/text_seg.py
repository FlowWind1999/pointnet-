import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import part_dataset_all_normal

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log_eval', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', type=int, default=9160, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

VOTE_NUM = 12

EPOCH_CNT = 0
result_file=r'./result.txt'

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
print(MODEL_PATH)
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
NUM_CLASSES = 2

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'ours')
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False,
                                                         split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print (is_training_pl)

            print ("--- Get model and loss")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()
            # saver=tf.train.import_meta_graph("log/model.ckpt.meta")

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        # saver.restore(sess, "log/model.ckpt")
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}
        eval_one_epoch(sess, ops)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps, normal, seg = dataset[idxs[i + start_idx]]
        batch_data[i, :, 0:3] = ps
        batch_data[i, :, 3:6] = normal
        batch_label[i, :] = seg
    return batch_data, batch_label


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET) + BATCH_SIZE - 1) / BATCH_SIZE

    seg_classes = TEST_DATASET.seg_classes

    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(1):
        if batch_idx % 20 == 0:
            log_string('%03d/%03d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx + 1) * BATCH_SIZE)
        cur_batch_size = end_idx - start_idx
        cur_batch_data, cur_batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        pred_val = np.zeros((BATCH_SIZE, NUM_POINT, NUM_CLASSES))
        for _ in range(VOTE_NUM):
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}
            temp_loss_val, temp_pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            # print(temp_pred_val.shape)
            pred_val += temp_pred_val
        # ---------------------------------------------------------------------
        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            print(len(cur_pred_val[0]))
            label = cur_pred_val[0]
            print (label)
            print(len(batch_data[0]))
            print(batch_data[0])
            points=[]
            for i in range(len(label)):
                point=batch_data[0][i]
                point=list(point)
                point=point[0:3]
                point.append(label[i])
                n_point=[]

                for i in point:
                    item=str(i)
                    n_point.append(item)

                n_point='    '.join(n_point)
                n_point=n_point+'\n'
                points.append(n_point)

            with open(result_file, 'w+') as f:
                f.writelines(points)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
