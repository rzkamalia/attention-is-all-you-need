import tensorflow as tf

def selector(input_text):
    split_text = tf.strings.split(input_text, '\t')
    return {'input_1': split_text[0:1], 'input_2': 'starttoken ' + split_text[1:2]}, split_text[1:2] + ' endtoken'

def separator(input_text):
    split_text = tf.strings.split(input_text, '\t')
    return split_text[0:1], 'starttoken ' + split_text[1:2] + ' endtoken'

class Scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(Scheduler, self).__init__()
        self.d_model = tf.cast(d_model, tf.float64)
        self.warmup_steps = tf.cast(warmup_steps, dtype = tf.float64)

    def __call__(self, step):
        step = tf.cast(step, dtype = tf.float64)
        return (self.d_model**(-0.5))*tf.math.minimum(step**(-0.5), step * (self.warmup_steps ** -1.5))