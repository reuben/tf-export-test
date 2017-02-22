import tensorflow as tf
import random
from os import path

lstm_cell = tf.contrib.rnn.BasicLSTMCell(64)

outputs, _ = tf.nn.dynamic_rnn(lstm_cell,
    dtype=tf.float32,
    inputs=tf.constant([[[1.,1.,1.]], [[1.,1.,0.]]]),
    sequence_length=tf.constant([3,2]))

pred = tf.add(outputs, tf.constant(0.), name="y")

saver = tf.train.Saver(tf.global_variables())

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    with open(path.join("graph-def", "graph.pb"), "wb") as fout:
        fout.write(sess.graph.as_graph_def().SerializeToString())
    print("Written graph.")
    
    sess.run(init)

    # Save variables in checkpoint
    save_path = saver.save(sess, path.join("checkpoints", "model.checkpoint"), write_meta_graph=False)
    print("Saved checkpoint to {}.".format(save_path))
