"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

class Discriminator(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    self._enc_batch1 = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens1 = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    # self._enc_padding_mask1 = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    self._enc_batch2 = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens2 = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    # self._enc_padding_mask2 = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')

    self._labels = tf.placeholder(tf.int64, [hps.batch_size, 1], name='labels')

  def _make_feed_dict(self, batch):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
    """
    feed_dict = {}
    feed_dict[self._enc_batch1] = batch.enc_batch1
    feed_dict[self._enc_lens1] = batch.enc_lens1
    # feed_dict[self._enc_padding_mask1] = batch.enc_padding_mask1
    feed_dict[self._enc_batch2] = batch.enc_batch2
    feed_dict[self._enc_lens2] = batch.enc_lens2
    # feed_dict[self._enc_padding_mask2] = batch.enc_padding_mask2
    feed_dict[self._labels] = batch.labels
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len, i):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder%d' %i):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st

  def _add_decoder(self, inputs):
    with tf.variable_scope('decoder'):
      cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
      decoder_outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
      return decoder_outputs

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_overall_model(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('overall_model'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs1 = tf.nn.embedding_lookup(embedding, self._enc_batch1)
        emb_enc_inputs2 = tf.nn.embedding_lookup(embedding, self._enc_batch2)

      # Add the encoder.
      enc_outputs1, _, _ = self._add_encoder(emb_enc_inputs1, self._enc_lens1, 1)
      enc_outputs2, _, _ = self._add_encoder(emb_enc_inputs2, self._enc_lens2, 2)
      enc_outputs = tf.concat(axis=1, values=[enc_outputs1, enc_outputs2])

      # Add the decoder
      # Get time-major for getting last step
      decoder_outputs = tf.transpose(self._add_decoder(enc_outputs), [1, 0, 2])
      last_step = tf.gather(decoder_outputs, int(decoder_outputs.get_shape[0]) - 1)
      logits = tf.contrib.layers.fully_connected(last_step, 1, activation_fn=None)

      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.cast(self._labels, tf.float32),
              logits=tf.cast(tf.reshape(logits, [hps.batch_size, 1]), tf.float32))

          tf.summary.scalar('loss', self._loss)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_overall_model()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)
