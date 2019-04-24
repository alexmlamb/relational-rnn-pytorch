import numpy

def copying_data(T, n_data, n_sequence, batch_size):
    seq = numpy.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = numpy.zeros((n_data, T-1))
    zeros2 = numpy.zeros((n_data, T))
    marker = 9 * numpy.ones((n_data, 1))
    zeros3 = numpy.zeros((n_data, n_sequence))

    x = numpy.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = numpy.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
    x = x.reshape(x.shape[0] // batch_size, batch_size, x.shape[1])
    y = y.reshape(y.shape[0] // batch_size, batch_size, y.shape[1])
    return x, y

T = 80
n_train = 640
n_test = 640
n_sequence = 10
batch_size = 64

if __name__ == "__main__":
    train_x, train_y = copying_data(T, n_train, n_sequence,batch_size)
    test_x, test_y = copying_data(T, n_test, n_sequence, batch_size)

    print('train x y shapes', train_x.shape)
    print(train_y.shape)

    for j in range(0,2):
        print(train_x[0,j])
        print(train_y[0,j])
    



