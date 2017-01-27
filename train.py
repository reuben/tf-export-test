import tensorflow as tf
import numpy as np
from os import path

BASEDIR = "/Users/Reuben/Development/tf-export-test"

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.get_variable("W", dtype=tf.float32, initializer=tf.random_uniform([1], -1.0, 1.0))
b = tf.get_variable("b", dtype=tf.float32, initializer=tf.zeros([1]))
y = tf.add(W * x_data, b, name="y")

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Add ops to save the variables.
saver = tf.train.Saver([W, b], sharded=True, write_version=2)

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
print("Training complete.")

# Learns best fit is W: [0.1], b: [0.3]

# Save variables in checkpoint
save_path = saver.save(sess, path.join(BASEDIR, "checkpoints", "model.checkpoint"))
print("Saved checkpoint to {}.".format(save_path))
