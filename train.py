from model import Generator, build_network
import tensorflow as tf
import logging

from utils.config import COMEDY_FULL_TEXT, BATCH_SIZE, SENTENCE_LENGTH

logging.basicConfig(format="%(levelname)s | %(asctime)s - %(message)s",
                    level=logging.DEBUG)

EPOCH = 100


def train():
    inputs, expect_tokens, loss = build_network()

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    generator = Generator(COMEDY_FULL_TEXT, BATCH_SIZE, SENTENCE_LENGTH)

    for epoch in range(EPOCH):
        while generator.have_next():
            sentences = generator.next_batch()
            following_tokens = generator.following_tokens()

            loss_value = sess.run(loss, feed_dict={
                inputs: sentences,
                expect_tokens: following_tokens
            })

            logging.info(f"Processed epoch: {epoch + 1}, batch: {generator.get_iter_count()} | loss: {loss_value}")

        generator.reset_iterator()


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    train()

    sess.close()

