__all__ = ['compat_tensorflow']

def _compat_tf_import(enable_gpu: bool = True):
    if not enable_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.get_logger().setLevel('ERROR')

    return tf

compat_tensorflow = _compat_tf_import()
