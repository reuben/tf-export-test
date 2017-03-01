#!/bin/bash

# Train model and save learned values
python train.py

# Freeze graph (embed checkpointed values as constant ops)
python ../tensorflow/tensorflow/python/tools/freeze_graph.py --input_graph=graph-def/graph.pb --input_binary --input_checkpoint=checkpoints/model.checkpoint --output_graph=graph-def/frozen_graph.pb --output_node_names=y

# Quantize graph
../tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=graph-def/frozen_graph.pb --out_graph=graph-def/quantized_graph.pb --outputs="y:0" --transforms="quantize_weights quantize_nodes"

# Import quantized graph and test it
python import.py
