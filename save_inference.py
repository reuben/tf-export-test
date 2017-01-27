import tensorflow as tf
from os import path

BASEDIR = "/Users/Reuben/Development/tf-export-test"

# Build inference graph
x = tf.placeholder(tf.float32, shape=[1], name="x")
W = tf.get_variable("W", shape=[1], dtype=tf.float32)
b = tf.get_variable("b", shape=[1], dtype=tf.float32)
y = tf.add(W * x, b, name="y")

# Add ops to restore the variables.
saver = tf.train.Saver([W, b], sharded=True, write_version=2)

with tf.Session() as sess:
    # Save variables in checkpoint
    saver.restore(sess, path.join(BASEDIR, "checkpoints", "model.checkpoint"))
    print("Restored variables.")

    with open(path.join(BASEDIR, "graph-def", "graph.pb"), "w") as fout:
        fout.write(sess.graph.as_graph_def().SerializeToString())

    print("Written graph def file.")
