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
    
    y = graph.get_tensor_by_name("y:0")
    
    sess = tf.Session()
    result = sess.run(y)
    print("result: {}".format(result))
