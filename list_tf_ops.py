import tensorflow as tf
from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format
    
def get_op_types(op):
    for attr in op.attr:
        if attr.type != 'type':
            continue
        return list(attr.allowed_values.list.type)
    return []

# directory where you did "git clone"
tensorflow_git_base = "tensorflow/"
ops_file = tensorflow_git_base+"/tensorflow/core/ops/ops.pbtxt"
ops = op_def_pb2.OpList()
text_format.Merge(open(ops_file).read(), ops)

for op in ops.op:
    # get templated string types
    if tf.string in get_op_types(op):
        print '{} => {} \n'.format(op.name, op.summary)
    for arg in op.input_arg:
        if arg.type == tf.string:
            print op.name, op.summary
            break
        

