from model import Generator, build_network
import tensorflow as tf
import logging


logging.basicConfig(format="%(levelname)s | %(asctime)s - %(message)s",
                    level=logging.DEBUG)


def train():
    inputs, expect_tokens, loss = build_network()

    sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver()
    # checkpoint = tf.train.get_checkpoint_state(CHECKOUT_FOLDER)
    generator = Generator()
    epoch = 1
    while generator.have_next():
        sentences = generator.next_batch()
        following_tokens = generator.following_tokens()

        loss_value = sess.run(loss, feed_dict={
            inputs: sentences,
            expect_tokens: following_tokens
        })

        logging.info(f"Processed epoch: {epoch + 1} | loss: {loss_value}")
        epoch += 1


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    train()

    sess.close()

