# encoding = utf8
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.contrib import rnn

from model.data_utils import iobes_iob, get_chunks
from model.utils import Progbar,test_ner
from model.data_utils import BatchManager



class Model(object):
    def __init__(self, sess, config, logger):

        self.logger = logger
        self.sess = sess
        self.config = config

        #lstm 层参数
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        #训练参数
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        # used = tf.sign(tf.abs(self.char_inputs))
        # length = tf.reduce_sum(used, reduction_indices=1)
        # self.lengths = tf.cast(length, tf.int32)

        self.lengths = tf.placeholder(dtype=tf.int32,
                                      name="Lengths")

        self.batch_size = tf.shape(self.char_inputs)[0]

        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # loss of the model
        self.loss = self.loss_layer()

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer,
                        trainable=config['train_embeddings'])
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, name=None):
        """Defines the loss"""
        with tf.variable_scope("crf_loss" if not name else name):
            if self.config['use_crf']:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                        self.logits, self.targets, self.lengths)
                self.trans_params = trans_params # need to evaluate it for decoding
                loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=self.targets)
                mask = tf.sequence_mask(self.lengths)
                losses = tf.boolean_mask(losses, mask)
                loss = tf.reduce_mean(losses)
            return loss

    # def loss_layer(self, project_logits, lengths, name=None):
    #     """
    #     calculate crf loss
    #     :param project_logits: [1, num_steps, num_tags]
    #     :return: scalar loss
    #     """
    #     with tf.variable_scope("crf_loss"if not name else name):
    #         small = -1000.0
    #         # pad logits for crf loss
    #         start_logits = tf.concat(
    #             [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
    #         pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
    #         logits = tf.concat([project_logits, pad_logits], axis=-1)
    #         logits = tf.concat([start_logits, logits], axis=1)
    #         targets = tf.concat(
    #             [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
    #
    #         self.trans = tf.get_variable(
    #             "transitions",
    #             shape=[self.num_tags + 1, self.num_tags + 1],
    #             initializer=self.initializer)
    #
    #         log_likelihood, self.trans = crf_log_likelihood(
    #             inputs=logits,
    #             tag_indices=targets,
    #             transition_params=self.trans,
    #             sequence_lengths=lengths+1)
    #         return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        chars, tags, lengths = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            # self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
            self.lengths: lengths,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, summary, _ = sess.run(
                [self.global_step, self.loss, self.merged, self.train_op],
                feed_dict)
            return global_step, loss, summary
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def _evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.minibatches():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        prog = Progbar(target=nbatches)

        train_manager = BatchManager(train, batch_size)


        # iterate over dataset
        for i, (words, labels) in enumerate(train_manager.minibatches()):
            words_padded, sequence_length = train_manager.pad_sequences(words)
            tags_padded, _ = train_manager.pad_sequences(labels)

            feed_dict = self.create_feed_dict(is_train=True, batch=(words_padded, tags_padded, sequence_length))

            global_step, batch_loss, summary, _ = self.sess.run(
                [self.global_step, self.loss, self.merged, self.train_op],
                feed_dict)

            # step, batch_loss, summary, _ = self.run_step(self.sess, is_train=True,
            #                                  batch=(words_padded, tags_padded, sequence_length))


            # print(batch_loss)
            prog.update(i+1, [("train_loss", batch_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)


        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary()  # tensorboard
        for epoch in range(100):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        100))

            score = self.run_epoch(train, dev, epoch)
            self.lr *= self.config['lr_decay']  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config['nepoch_no_imprv']:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break

    def evaluate(self, name, data, id_to_tag, logger):
        logger.info("evaluate:{}".format(name))
        ner_results = self._evaluate(self.sess, data, id_to_tag)
        eval_lines = test_ner(ner_results, self.config['result_path'])
        for line in eval_lines:
            logger.info(line)
        f1 = float(eval_lines[1].strip().split()[-1])

        if name == "dev":
            best_test_f1 = self.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.best_dev_f1, f1).eval()
                logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1
        elif name == "test":
            best_test_f1 = self.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.best_test_f1, f1).eval()
                logger.info("new best test f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1
    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config['ckpt_path'], self.sess.graph)

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config['ckpt_path']):
            os.makedirs(self.config['ckpt_path'])
        self.saver.save(self.sess, self.config['ckpt_path'])


    def predict_batch(self, fd):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """

        # fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config['use_crf']:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, self.lengths):
                logit = logit[:sequence_length]    # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        test_manager = BatchManager(test, self.batch_size)

        for words, labels in test_manager.minibatches():

            words_padded, sequence_lengths = test_manager.pad_sequences(words)

            feed_dict = self.create_feed_dict(is_train=False, batch=(words_padded, None, sequence_lengths))

            labels_pred = self.predict_batch(feed_dict)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config['tag_to_id']))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config['tag_to_id']))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}