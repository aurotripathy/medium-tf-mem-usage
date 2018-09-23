"""
Yaroslav Butalov's solution
https://stackoverflow.com/questions/40190510/tensorflow-how-to-log-gpu-memory-vram-utilization

Ensure that the following is in your path:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/"
"""
import tensorflow as tf

no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0,
                             do_common_subexpression_elimination=False,
                             do_function_inlining=False,
                             do_constant_folding=False)
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        log_device_placement=True, allow_soft_placement=False,
                        device_count={"CPU": 3},
                        inter_op_parallelism_threads=3,
                        intra_op_parallelism_threads=1)
sess = tf.Session(config=config)

with tf.device("gpu:0"):
    a = tf.ones((13, 1))
with tf.device("gpu:1"):
    b = tf.ones((1, 13))
with tf.device("gpu:2"):
    c = a+b

sess = tf.Session(config=config)
run_metadata = tf.RunMetadata()
sess.run(c,
         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                               output_partition_graphs=True),
         run_metadata=run_metadata)
with open("/tmp/run2.txt", "w") as out:
    out.write(str(run_metadata))
