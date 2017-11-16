import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

def load_charset(charset_file):
    return ''.join([line.rstrip('\n') for line in open(charset_file)])

def freeze(input_graph_path, checkpoint_path, output_node_names, input_saver_def_path="", input_binary=False,
           restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
           output_frozen_graph_name="frozen_output.pb",
           clear_devices=True):
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                              checkpoint_path, output_node_names, restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")


def optimize_graph(graph_path, input_nodes, output_nodes):
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(graph_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    input_graph_def = tf.graph_util.remove_training_nodes(input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_nodes,
        output_nodes,
        tf.float32.as_datatype_enum
    )

    f = tf.gfile.FastGFile("optimized_" + graph_path, "w")
    f.write(output_graph_def.SerializeToString())
