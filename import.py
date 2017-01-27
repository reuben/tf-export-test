import tensorflow as tf
from os import path

BASEDIR = "/Users/Reuben/Development/tf-export-test"

input_graphdef = path.join(BASEDIR, "graph-def", "quantized_graph.pb")

graph = tf.Graph()
with graph.as_default():
    pb = tf.GraphDef()
    with open(input_graphdef, "rb") as fin:
        # pb.MergeFromString(fin.read()) # text
        pb.ParseFromString(fin.read()) # binary
    imports = tf.import_graph_def(pb, name="")
    
    # print("Ops: {}".format([op.name for op in graph.get_operations()]))

    y = graph.get_tensor_by_name("y:0")
    
    sess = tf.Session()
    print("23 * 0.1 + 0.3 = {}".format(sess.run(y, feed_dict={"x:0": [23]})))
