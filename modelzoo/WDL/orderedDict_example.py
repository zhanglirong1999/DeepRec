from pyexpat import features
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import collections
import random
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
def getData():
    index = []
    for i in range(0,2000000):
        index.append(random.randint(0,10000))
    data = np.array(index)
    return data
features = collections.OrderedDict()
for col in FEATURE_COLUMNS:
    features[col] = getData()
labels = np.random.rand(2000000)
start = time.time()
dataset = tf.data.Dataset.from_tensor_slices((features,labels))

dataset = dataset.batch(100)
dataset = dataset.prefetch(2)
# print(f'cost run time:{time.time() - t2:.8f}s')
iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
# iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
train_init_op = iterator.make_initializer(dataset)
global_step = tf.train.get_or_create_global_step()
sess_config = tf.ConfigProto()
scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), train_init_op))
stop_hook = tf.train.StopAtStepHook(last_step=1000)
hooks = []
hooks.append(stop_hook)
tm = time.time()
s = str(int(time.time()))
checkpoint_dir = os.path.join('./result/'+s)
with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            hooks=hooks,
            checkpoint_dir = checkpoint_dir,
            save_checkpoint_steps=1000,
            summary_dir= None,
            save_summaries_steps=1000,
            config=sess_config) as sess:
    print(f'cost session create:{time.time() - tm:.8f}s')
    t2 = time.time()
    for i in range(1,100):
        sess.run(next_element)
    print(f'cost run time:{time.time() - t2:.8f}s')
    print('done')
