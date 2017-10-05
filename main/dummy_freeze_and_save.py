from main.utils import freeze, optimize_graph

frozen_graph_path="frozen_bi_lstm_ctc_ocr.pb"
freeze(
    input_graph_path="log/train/bi_lstm_ctc_ocr.pbtxt",
    checkpoint_path="checkpoint/ocr-model-0.ckpt-0",
    output_node_names="output",
    input_saver_def_path="",
    output_frozen_graph_name=frozen_graph_path,
    input_binary=False
)
optimize_graph(frozen_graph_path, ["input", "seq_len_input"], ["output"])
