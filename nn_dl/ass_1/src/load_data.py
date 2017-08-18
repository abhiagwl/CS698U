import cPickle
import gzip
import numpy as np

def load_():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = cPickle.load(f)
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)


    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
