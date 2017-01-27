```bash
# Train model and save learned values
python train.py
# Restore learned values and export simpler inference graph
python save_inference.py
# Freeze inference graph (embed checkpointed values as constant ops)
python ../tensorflow/tensorflow/python/tools/freeze_graph.py --input_graph=graph-def/graph.pb --input_binary --input_checkpoint=checkpoints/model.checkpoint --output_graph=graph-def/frozen_graph.pb --output_node_names=y
# Quantize graph
python ../tensorflow/tensorflow/tools/quantization/quantize_graph.py --input=graph-def/frozen_graph.pb --output_node_names=y --output=graph-def/quantized_graph.pb --mode=eightbit
# Import quantized graph and test it
python import.py
```
