# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict
import argparse
import tensorflow as tf
import numpy as np
from model.loder import LoadDataset
from model.loder import load_vocab, load_word2vec

from model.model import Model
from model.utils import get_logger,make_path,save_config,load_config
from model.utils import create_model
from model.data_utils import get_processing_word

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument("--clean",       default=True,  type=bool,   help="clean train folder")
parser.add_argument("--train",       default=True,  type=bool,    help="Wither train the model")

# configurations for file
parser.add_argument("--train_file", default="./data/dg_train.txt", type=str, help="Path for train data")
parser.add_argument("--dev_file", default='./data/dg_dev.txt', type=str, help="Path for dev data")
parser.add_argument("--test_file", default='./data/dg_test.txt', type=str, help="Path for test data")
parser.add_argument("--emb_file", default='./data/word2vec.trimmed.npz', type=str, help="Path for emb data")

parser.add_argument("--ckpt_path",    default="ckpt",    type=str,  help="Path to save model")
parser.add_argument("--summary_path", default="summary",  type=str,    help="Path to store summaries")
parser.add_argument("--log_file",     default="train.log", type=str,   help="File for log")
parser.add_argument("--map_file",     default="maps.pkl",   type=str,  help="file for maps")
parser.add_argument("--vocab_file",   default="./data/words.txt",  type=str, help="File for vocab")
parser.add_argument("--tag_file",   default="./data/tags.txt",  type=str, help="File for tags")
parser.add_argument("--config_file",  default="config_file",  type=str, help="File for config")
parser.add_argument("--script",       default="conlleval",    type=str, help="evaluation script")
parser.add_argument("--result_path",  default="result",       type=str, help="Path for results")

# configurations for model
parser.add_argument("--seg_dim",   default=0, type=int, help="Embedding size for segmentation, 0 if not used")
parser.add_argument("--char_dim",  default=300, type=int, help="Embedding size for characters")
parser.add_argument("--lstm_dim",  default=300,  type=int, help="Num of hidden units in LSTM")
parser.add_argument("--tag_schema", default="iob", type=str, help="tagging schema iobes or iob")

# configurations for training
parser.add_argument("--clip",       default=5,      type=float,       help="Gradient clip")
parser.add_argument("--dropout",    default=0.5,    type=float,       help="Dropout rate")
parser.add_argument("--batch_size", default=20,     type=int,       help="batch size")
parser.add_argument("--lr",         default=0.001,  type=float,      help="Initial learning rate")
parser.add_argument("--lr_decay",   default=0.9,  type=float,      help="Initial learning rate decay")
parser.add_argument("--nepoch_no_imprv",   default=5,  type=int,      help="number of epoch not improvement")
parser.add_argument("--use_crf",   default=True,  type=bool,      help="whether use crf")
parser.add_argument("--optimizer",  default="adam", type=str,     help="Optimizer for training")
parser.add_argument("--pre_emb",    default=True,   type=bool,    help="Wither use pre-trained embedding")
parser.add_argument("--zeros",      default=False,  type=bool,    help="Wither replace digits with zero")
parser.add_argument("--lower",      default=False,  type=bool,     help="Wither lower case")
parser.add_argument("--max_epoch",   default=100,   type=int, help="maximum training epochs")
parser.add_argument("--steps_check", default=100,   type=int, help="steps per checkpoint")
parser.add_argument("--train_embeddings", default=True,   type=bool, help="if the embeddings is trainable")

args = parser.parse_args()
assert args.clip < 5.1, "gradient clip should't be too much"
assert 0 <= args.dropout < 1, "dropout rate between 0 and 1"
assert args.lr > 0, "learning rate must larger than zero"
assert args.optimizer in ["adam", "sgd", "adagrad"]


# config for the model

def config_model(char_to_id, tag_to_id, id_to_tag):
    config = OrderedDict()
    config['log_file'] = args.log_file
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = args.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = args.seg_dim
    config["lstm_dim"] = args.lstm_dim
    config["batch_size"] = args.batch_size

    config["emb_file"] = args.emb_file
    config["clip"] = args.clip
    config["dropout_keep"] = 1.0 - args.dropout
    config["optimizer"] = args.optimizer
    config["lr"] = args.lr
    config['lr_decay'] = args.lr_decay
    config['nepoch_no_imprv'] = args.nepoch_no_imprv
    config['use_crf'] = args.use_crf
    config["tag_schema"] = args.tag_schema
    config["pre_emb"] = args.pre_emb
    config["zeros"] = args.zeros
    config["lower"] = args.lower
    config["train_embeddings"] = args.train_embeddings
    config["ckpt_path"] = args.ckpt_path
    config["tag_to_id"] = tag_to_id
    config["id_to_tag"] = id_to_tag
    config["result_path"] = args.result_path
    return config


# def evaluate(sess, model, name, data, id_to_tag, logger):
#     logger.info("evaluate:{}".format(name))
#     ner_results = model.evaluate(sess, data, id_to_tag)
#     eval_lines = test_ner(ner_results, FLAGS.result_path)
#     for line in eval_lines:
#         logger.info(line)
#     f1 = float(eval_lines[1].strip().split()[-1])
#
#     if name == "dev":
#         best_test_f1 = model.best_dev_f1.eval()
#         if f1 > best_test_f1:
#             tf.assign(model.best_dev_f1, f1).eval()
#             logger.info("new best dev f1 score:{:>.3f}".format(f1))
#         return f1 > best_test_f1
#     elif name == "test":
#         best_test_f1 = model.best_test_f1.eval()
#         if f1 > best_test_f1:
#             tf.assign(model.best_test_f1, f1).eval()
#             logger.info("new best test f1 score:{:>.3f}".format(f1))
#         return f1 > best_test_f1


def train():

    word_to_id, id_to_word = load_vocab(args.vocab_file)
    tag_to_id, id_to_tag = load_vocab(args.tag_file)
    processing_word = get_processing_word(word_to_id)
    processing_tag = get_processing_word(tag_to_id, allow_unk=False)

    # load data sets
    train_sentences = LoadDataset(args.train_file,processing_word,processing_tag)
    dev_sentences = LoadDataset(args.dev_file, processing_word,processing_tag)
    test_sentences = LoadDataset(args.test_file,processing_word,processing_tag)

    # Use selected tagging scheme (IOB / IOBES)
    # update_tag_scheme(train_sentences, args.tag_schema)
    # update_tag_scheme(test_sentences, args.tag_schema)

    if os.path.isfile(args.config_file):
        config = load_config(args.config_file)
    else:
        config = config_model(word_to_id, tag_to_id, id_to_tag)
        save_config(config, args.config_file)

    make_path(args)
    log_path = os.path.join("log", args.log_file)
    logger = get_logger(log_path)

    with tf.Session() as sess:

        model = create_model(sess, Model, args.ckpt_path, load_word2vec, config, logger)

        model.train(train_sentences, dev_sentences)

if __name__ == '__main__':
    train()