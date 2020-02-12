from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import hashlib
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

data_index = 0

def generate_batch(skip_window, batch_size, num_skips, data):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index - span) % len(data)
    return batch, labels

def serializeInputMatrix(data,sw):
    input_matrix = []
    for x in range (len(data[0])):
        for y in range(len(data[0][x])):
            input_matrix.append(data[0][x][y])
        input_matrix.append(data[1][x])
        for y in range(sw):
            input_matrix.append('#')
    
    return input_matrix

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {word: index for index, (word, _) in enumerate(count)}
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def train_graph(data, reverse_dictionary, batch_size, embedding_size, num_sampled, num_skips, skip_window, vocabulary_size, log_dir):
  graph = tf.Graph()
  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      with tf.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
      with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
      with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss and why choosing NCE over tf.nn.sampled_softmax_loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    #   http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  # Step 5: Begin training.
  num_steps = 100001
  with tf.compat.v1.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # Define metadata variable.
      run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned
      # "summary" variable. Feed metadata variable to session for visualizing
      # the graph in TensorBoard.
      _, summary, loss_val = session.run([optimizer, merged, loss],
                                          feed_dict=feed_dict,
                                          run_metadata=run_metadata)
      average_loss += loss_val

      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word

          print(
              log_str,
              ', '.join([reverse_dictionary[nearest[k]] for k in range(top_k)]))
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

      # Save the model for checkpoints.
      saver.save(session, os.path.join(log_dir, 'model.ckpt'))

      # Create a configuration for visualizing embeddings with the labels in
      # TensorBoard.
      config = projector.ProjectorConfig()
      embedding_conf = config.embeddings.add()
      embedding_conf.tensor_name = embeddings.name
      embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
      projector.visualize_embeddings(writer, config)

    writer.close()
    return graph, init, normalized_embeddings,saver
  