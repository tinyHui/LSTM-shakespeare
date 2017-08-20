from model import Generator, build_network
import tensorflow as tf
import logging
import os

from utils.config import COMEDY_FULL_TEXT, BATCH_SIZE, SENTENCE_LENGTH, Mode

tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'shakespeare', 'model save prefix.')
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(format="%(levelname)s | %(asctime)s - %(message)s",
                    level=logging.DEBUG)

EPOCH = 100


def train():
    inputs, expect_tokens, last_state, loss, train_op = build_network(mode=Mode.TRAIN)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    generator = Generator(COMEDY_FULL_TEXT, BATCH_SIZE, SENTENCE_LENGTH)

    saver = tf.train.Saver(tf.global_variables())

    start_epoch = 0
    epoch = 0

    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
    if checkpoint:
        logging.info("Found checkoutpoint, trying to restore from it")
        saver.restore(sess, checkpoint)
        logging.info("Restored successfully")
        start_epoch += int(checkpoint.split('-')[-1])

    logging.info('start training')
    try:
        for epoch in range(start_epoch, EPOCH):
            while generator.have_next():
                sentences = generator.next_batch()
                following_tokens = generator.following_tokens()

                loss_value, _, _ = sess.run([loss, last_state, train_op], feed_dict={
                    inputs: sentences,
                    expect_tokens: following_tokens
                })

                logging.info(f"Processed epoch: {epoch + 1}, batch: {generator.get_iter_count()} | loss: {loss_value}")

            generator.reset_iterator()

            if epoch % 5 == 0:
                logging.info(f"Model saved in epoch {epoch}")
                saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
    except KeyboardInterrupt:
        logging.info("Interrupted, try saving current training as a checkpoint")
        saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        logging.info(f"Last epoch were saved, next time will start from epoch {epoch}")


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    train()
    sess.close()

