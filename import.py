# coding: utf-8
import tensorflow as tf
from os import path

input_graphdef = path.join("graph-def", "quantized_graph.pb")

graph = tf.Graph()
with graph.as_default():
    pb = tf.GraphDef()
    with open(input_graphdef, "rb") as fin:
        # text_format.Merge(fin.read(), pb) # text
        pb.ParseFromString(fin.read()) # binary
    imports = tf.import_graph_def(pb, name="")
    
    outputs = graph.get_tensor_by_name("y:0")

    sess = tf.Session()
    result = sess.run(outputs)
    print("result: {}".format(result))
