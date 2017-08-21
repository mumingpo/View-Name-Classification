import json
import tensorflow as tf
from collections import Counter
from itertools import dropwhile
import numpy as np
from datetime import datetime

S = {
    'bs': 50,               #batch size
    'ti': 2000,             #training iterations
    'lr': 1e-2,             #learning rate
    'il': 3,                #interest level, >= which would be deemed interesting
    'tc': 2000,             #number of test cases at the end
    }
tf.set_random_seed(0)
np.random.seed(0)

def read_data():
    entries = []
    with open(r'C:\Users\mumin\Documents\Visual Studio 2017\Projects\view_name_classification\view_name_classification\trains.jsonl', 'r') as f:
        for line in f:
            s = line.strip('\x00')
            entries.append(json.loads(s[s.rfind('\x00')+1:]))
    return entries

#view_names = dict()
#for i in entries:
#  vndir = i['view_name'].split('.')
#  h = view_names
#  for j in vndir:
#    if j not in h:
#      h[j] = dict()
#    h = h[j]



def main():
    entries = read_data()

    view_names = set()
    interesting_attributes = Counter()

    for i in entries:
        view_names.add(i['view_name'])
        pathdir = i['path'].split('/')[1:]
        for j in pathdir:
            interesting_attributes[j] += 1

    for key, count in dropwhile(lambda key_count: key_count[1] >= S['il'], interesting_attributes.most_common()):
        del interesting_attributes[key]

    view_names = list(view_names)
    i2n, n2i = dict(), dict()
    for i, n in enumerate(view_names):
        i2n[i] = n
        n2i[n] = i

    interesting_attributes = interesting_attributes.keys()
    i2a, a2i = dict(), dict()
    for i, a in enumerate(interesting_attributes):
        i2a[i] = a
        a2i[a] = i

    global_step= tf.Variable(0., False, dtype=tf.float32)

    x = tf.placeholder(tf.float32, [None, len(interesting_attributes)], name = 'input')
    y_ = tf.placeholder(tf.float32, [None, len(view_names)], name = 'label')
    W = tf.Variable(tf.random_normal([len(interesting_attributes), len(view_names)]), dtype = tf.float32)
    b = tf.Variable(tf.random_normal([len(view_names)], dtype = tf.float32))
    linear = tf.matmul(x, W) + b
    pred = tf.argmax(tf.nn.softmax(linear), 1)
    actual = tf.argmax(y_, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))
    loss = tf.losses.softmax_cross_entropy(y_, linear)
    learning_rate = tf.train.exponential_decay(S['lr'], global_step, 500, 0.5, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        tf.set_random_seed(0)
        sess.run(init)
        for i in range(S['ti']):
            feed_indices = np.random.randint(low = 0, high = len(entries), size = S['bs'])
            feed_input_arrays = []
            feed_labels = np.array([[0.] * len(view_names) for _ in range(S['bs'])], dtype = np.float32)
            for ji, j in enumerate(feed_indices):
                ia = np.zeros(len(interesting_attributes), dtype = np.float32)
                feed_labels[ji][n2i[entries[j]['view_name']]] = 1.
                for k in entries[j]['path'].split('/')[1:]:
                    if k in interesting_attributes:
                        ia[a2i[k]] = 1.
                feed_input_arrays.append(ia)
            feed_input = np.stack(feed_input_arrays)

            l, a, _ = sess.run((loss, accuracy, train), feed_dict={x: feed_input, y_: feed_labels})
            print('batch {:4}/{}: accuracy = {:.2f}%, loss = {}'.format(i+1, S['ti'], a*100, l))

        test_indices = np.random.randint(low = 0, high = len(entries), size = S['tc'])
        test_input_arrays = []
        test_labels = np.array([[0.] * len(view_names) for _ in range(S['tc'])], dtype = np.float32)
        for ii, i in enumerate(test_indices):
            ia = np.zeros(len(interesting_attributes), dtype = np.float32)
            test_labels[ii][n2i[entries[i]['view_name']]] = 1
            for j in entries[i]['path'].split('/')[1:]:
                if j in interesting_attributes:
                    ia[a2i[j]] = 1
            test_input_arrays.append(ia)
        test_input = np.stack(test_input_arrays)
        
        test_begin = datetime.now()
        test_p, test_a, acc, los = sess.run((pred, actual, accuracy, loss), feed_dict={x: test_input, y_: test_labels})
        test_end = datetime.now()
        test_elapse = test_end - test_begin
        print('{} tests completed in {} seconds\n  Accuracy: {:.2f}%\n  Loss: {}\n\n\n'.format(S['tc'], test_elapse.total_seconds(), acc*100, los))

        for ti, (tp, ta) in enumerate(zip(test_p, test_a)):
            if tp != ta:
                print('Mismatch:\n    Path: {}\n    Should obtain {},\n    got {}'.format(entries[test_indices[ti]]['path'], i2n[ta], i2n[tp]))


if __name__ == '__main__':
    main()