#coding=utf8
import datetime
import gc
import time
import os
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

from model.DPAM_model import DPAM
from data_helper import *

BasePath = sys.path[0]

# Parameters
tf.flags.DEFINE_float("dev_sample_percentage", 0.10, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("burning_step", 1500, "the burning step in the training process")
tf.flags.DEFINE_boolean("shuffle_data", False, "Allow device soft device placement")

# Model Hyperparameters
tf.flags.DEFINE_integer("my_embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0005, "L2 regularizaion lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


topn = 70

#### print Parameters
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

train_dev_x_file_path = BasePath + "/new_data/train_dev_x.txt"
train_dev_y_file_path = BasePath + "/new_data/train_dev_y.txt"
train_dev_rule_file_path = BasePath + "/new_data/train_dev_rule.txt"
laws_dict_path = "./new_data/one_hot_vocab_70.txt"

x_text, y = get_dev_train_data(train_dev_x_file_path, train_dev_y_file_path)
rule_input = get_rule_data(train_dev_rule_file_path)
laws_dict = load_laws_dict(laws_dict_path)

min_frequence = 10
average_document_length = 1000
vocab_processor = learn.preprocessing.VocabularyProcessor(average_document_length, min_frequency=min_frequence)
x = np.array(list(vocab_processor.transform(x_text)))
x_rule = np.array(list(vocab_processor.transform(rule_input)))

print("make embedding vocab finished")
print("average_document_length is: " + str(average_document_length))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))


### shuffle dataset
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
if FLAGS.shuffle_data:
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
else:
    x_shuffled = x
    y_shuffled = y

### release memory
del x
del x_text
del y
gc.collect()
print("finished free memory")

# split train and dev data
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

default_device = tf.device('/gpu:0')
with default_device, tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
    session_conf.gpu_options.allow_growth = False

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = DPAM(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.my_embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        cnn.add_placeholders()
        cnn.add_embedding()
        cnn.add_conv_pool()
        cnn.add_dropout()
        cnn.add_output()
        cnn.add_ratio_cal()
        cnn.add_loss()

        global_step = tf.Variable(0, name = "global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

        optimizer2 = tf.train.AdamOptimizer(1e-3)
        grads_and_vars2 = optimizer2.compute_gradients(cnn.start_loss)
        train_op2 = optimizer2.apply_gradients(grads_and_vars2, global_step = global_step)


        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        sess.run(tf.global_variables_initializer())

        # train_step
        def train_step(x_batch, y_batch, x_rule):
            """
                A single training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                cnn.input_x : x_batch,
                cnn.input_y : y_batch,
                cnn.input_rule : x_rule,
                cnn.dropout_keep_prob : FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, y_predictions = sess.run(
                [train_op, global_step, train_summary_op, cnn.cross_loss, cnn.predictions],
                feed_dict)

            assert len(y_predictions) == len(y_batch), "the size of y_pred and y_true is different"

            precision = metrics.precision_score(np.array(y_batch), y_predictions, average='samples')
            recall = metrics.recall_score(np.array(y_batch), y_predictions, average='samples')
            f1_score = metrics.f1_score(np.array(y_batch), y_predictions, average='samples')

            precision_macro = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
            recall_macro = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
            f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')

            precision_micro = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
            recall_micro = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
            f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')

            hamming_loss = metrics.hamming_loss(np.array(y_batch), y_predictions)
            jaccard = metrics.jaccard_similarity_score(np.array(y_batch), y_predictions)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
            print("{}: step {}, jaccard similarity score is: {}".format(time_str, step, jaccard))

            train_summary_writer.add_summary(summaries, step)
            return [precision, recall, f1_score,
                    precision_macro, recall_macro, f1_score_macro,
                    precision_micro, recall_micro, f1_score_micro,
                    hamming_loss, jaccard]

        # train_step
        def train_step2(x_batch, y_batch, x_rule):
            """
                A single training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_rule: x_rule,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, y_predictions = sess.run(
                [train_op, global_step, train_summary_op, cnn.start_loss, cnn.predictions],
                feed_dict)

            assert len(y_predictions) == len(y_batch), "the size of y_pred and y_true is different"

            precision = metrics.precision_score(np.array(y_batch), y_predictions, average='samples')
            recall = metrics.recall_score(np.array(y_batch), y_predictions, average='samples')
            f1_score = metrics.f1_score(np.array(y_batch), y_predictions, average='samples')

            precision_macro = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
            recall_macro = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
            f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')

            precision_micro = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
            recall_micro = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
            f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
            hamming_loss = metrics.hamming_loss(np.array(y_batch), y_predictions)
            jaccard = metrics.jaccard_similarity_score(np.array(y_batch), y_predictions)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
            print("{}: step {}, jaccard similarity score is: {}".format(time_str, step, jaccard))
            train_summary_writer.add_summary(summaries, step)

            return [precision, recall, f1_score,
                    precision_macro, recall_macro, f1_score_macro,
                    precision_micro, recall_micro, f1_score_micro,
                    hamming_loss, jaccard]


        def dev_step(x_batch, y_batch, x_rule, writer = None):
            feed_dict = {
                cnn.input_x : x_batch,
                cnn.input_y : y_batch,
                cnn.input_rule : x_rule,
                cnn.dropout_keep_prob : 1
            }
            step, summaries, loss, y_predictions= sess.run(
                [global_step, dev_summary_op, cnn.dev_loss, cnn.predictions], feed_dict)

            assert len(y_predictions) == len(y_batch), "the size if y_pred and y_true is different"
            precision = metrics.precision_score(np.array(y_batch), y_predictions, average='samples')
            recall = metrics.recall_score(np.array(y_batch), y_predictions, average='samples')
            f1_score = metrics.f1_score(np.array(y_batch), y_predictions, average='samples')

            precision_macro = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
            recall_macro = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
            f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')

            precision_micro = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
            recall_micro = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
            f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
            hamming_loss = metrics.hamming_loss(np.array(y_batch), y_predictions)
            jaccard = metrics.jaccard_similarity_score(np.array(y_batch), y_predictions)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
            print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                    .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
            print("{}: step {}, jaccard similarity score is: {}".format(time_str, step, jaccard))

            return y_predictions, np.array(y_batch), [precision, recall, f1_score,
                                                      precision_macro, recall_macro,
                                                      f1_score_macro,
                                                      precision_micro, recall_micro,
                                                      f1_score_micro,
                                                      hamming_loss, jaccard]

        #####  train and dev
        evaluate_every = 50
        checkpoint_every = 50
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        i = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            if(len(x_batch) != 64):
                continue
            if(i < FLAGS.burning_step):
                train_merge_result = train_step2(x_batch, y_batch, x_rule)
                current_step = tf.train.global_step(sess, global_step)
            else:
                train_merge_result = train_step(x_batch, y_batch, x_rule)
                current_step = tf.train.global_step(sess, global_step)
            i += 1
            #
            # step_list.append(current_step)
            # train_result_list.append(train_merge_result)

            if current_step % evaluate_every == 0:
                print("\n Evaluation: ")
                print("###########################################")
                y_predictions, y_batch, dev_merge_result = dev_step(x_dev, y_dev, x_rule, writer = dev_summary_writer)
                print("###########################################")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {} in the step {}.\n".format(path, current_step))
            if current_step == 8001:
                break






