import tensorflow as tf
from os import path, getcwd

BASEDIR = getcwd()

input_graphdef = path.join(BASEDIR, "graph-def", "quantized_graph.pb")

graph = tf.Graph()
with graph.as_default():
    pb = tf.GraphDef()
    with open(input_graphdef, "rb") as fin:
        # pb.MergeFromString(fin.read()) # text
        pb.ParseFromString(fin.read()) # binary
    imports = tf.import_graph_def(pb, name="")
    
    x = graph.get_tensor_by_name("x:0")
    x_len = graph.get_tensor_by_name("x_len:0")
    y = graph.get_tensor_by_name("y:0")
    
    sess = tf.Session()
    pred = sess.run(y, feed_dict={
        x: [[[2.0,2.1,2.2,2.3,2.4,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]],
        x_len: [5]
    })
    print("prediction: {}".format("linear" if pred == 0 else "random"))
