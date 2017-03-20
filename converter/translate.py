# These are some imports used by tensorflow contributors. 
# We're not completely sure why, but lets use it! must be good stuff
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange    # pylint: disable=redefined-builtin

import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf

import nmt_model
import seq2seq_model
import nltk

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                                                    "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                                                    "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                                                        "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")

tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("beam_search", False, "Set to use beam_search for interactive decoding.")
tf.app.flags.DEFINE_integer("beam_size", 3, "Number of layers in the model.")

tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,"Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(10, 15), (40,50)]

# Default Symbols required for our seq2seq model
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]

    with open(source_path, 'r') as source_file:
        with open(target_path, 'r') as target_file:

            source_line, target_line = source_file.readline(), target_file.readline()

            while source_line and target_line:

                source_ids = [int(x) for x in source_line.split()]
                target_ids = [int(x) for x in target_line.split()]

                target_ids.append(EOS_ID)

                # Find the appropriate bucket for this source, target pair 
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break

                source_line, target_line = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    import nmt_model

    model = nmt_model.NMT_Model(
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size,
            _buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            forward_only=forward_only,
            dtype=dtype,
            beam_search=FLAGS.beam_search,
            beam_size=FLAGS.beam_size)

    # try loading embeding matrix
    encoder_mtrx = np.load('trans.npy')
    decoder_mtrx = np.load('orig.npy')

    if encoder_mtrx is not None or decoder_mtrx is not None:
        model.define_embedding_mtrx(encoder_mtrx, decoder_mtrx)

    model.define_loss_func()
    model.define_nmt_cell(FLAGS.size)

    model.define_nmt_seq_func("attention")
    model.define_nmt_buckets(_buckets)
    model.define_train_ops()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Create model from checkpoint")
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created a new model. ")
        session.run(tf.global_variables_initializer())

    return model

def train():
    from_train = './data/all_modern.snt.aligned.ids'
    to_train = './data/all_original.snt.aligned.ids'
    from_dev = from_train
    to_dev = to_train

    with tf.Session() as sess:
        # Create model.
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
                     % FLAGS.max_train_data_size)
        dev_set = read_data(from_dev, to_dev)
        train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                                     for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                                     target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Change learning rate accordingly
            if current_step - 4000 >= 0 and current_step % 1000 == 0: sess.run(model.learning_rate_decay_op)

            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.8f step-time %.2f perplexity "
                             "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                 step_time, perplexity))

                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

def tokenize(sentence):
    result = []
    arr = nltk.word_tokenize(sentence)
    for word in arr:
        new_arr = word.strip().aplit()
        result.append(new_arr)
    return result

def lookup_tokens(file_name):
    dic = dict()
    with open(file_name, 'r') as f:
        for line in f:
            dic[line.strip('\n')] = len(dic.keys())
        f.close()
    return dic

def rev_lookup_tokens(file_name):
    dic = dict()
    with open(file_name, 'r') as f:
        for line in f:
		    dic[len(dic.keys())] = line.strip('\n')
        f.close()
	return dic

def sentence_to_tokens(sentence, vocab):
    sentence = tokenize(sentence)
    result = []
    for word in sentence:
        result.append(vocab[word])
    return result

def get_bucket_id(token_ids):
    length = len(token_ids)
    result = length - 1

    for i, curr_bucket in enumerate(_buckets):
        if curr_bucket[0] >= length:
            result = i
            break

    return result

def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1 # set the batch size of our model to be 1

        from_vocab_path = "./data/rap.trans.aligned.tokens"
        to_vocab_path = "./data/rap.original.aligned.tokens"
        from_vocab = lookup_tokens(from_vocab_path)
        rev_to_vocab = rev_lookup_tokens(to_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        # Keep decoding as long as the user is keep feeding decoding inputs
        while sentence is not None:
            token_ids = sentence_to_tokens(sentence, from_vocab)

            bucket_id = get_bucket_id(token_ids)

            # Create a batch of size 1 out of the given tokens
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

            # Try decoding the given token_ids
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            # Beamsearch returns logits for the best symbols
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # Try to cut sentence when its supposed to end
            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]

            # Print out French sentence corresponding to outputs.
            print(" ".join([str(rev_to_vocab[output]) for output in outputs]))
            print("> ", end="")

            sys.stdout.flush()

            # Read the next sentence
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
