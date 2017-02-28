# coding: utf-8
import tensorflow as tf
from os import path

cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)

outputs, _ = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float32,
    sequence_length=tf.constant([3, 2]),
    inputs=tf.constant([[[1.,1.,1.]], [[1.,1.,0.]]]))

outputs = tf.identity(outputs, name="y")

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    with open(path.join("graph-def", "graph.pb"), "wb") as fout:
        fout.write(sess.graph.as_graph_def().SerializeToString())
    print("Written graph.")

    sess.run(tf.global_variables_initializer())
    sess.run(outputs)
    
    # Save variables in checkpoint
    save_path = saver.save(sess, path.join("checkpoints", "model.checkpoint"))
    print("Saved checkpoint to {}.".format(save_path))
