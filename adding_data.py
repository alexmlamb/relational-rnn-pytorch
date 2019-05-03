import numpy as np

def _adding_data(T):
    x_values = np.random.uniform(low=0, high=1, size=T)
    x_indicator = np.zeros(T, dtype=np.bool)
    x_indicator[np.random.randint(T/2)] = 1
    x_indicator[np.random.randint(T/2, T)] = 1
    #x = np.array(list(zip(x_values, x_indicator)))[np.newaxis]
    x = np.vstack((x_values, x_indicator)).T
    #y = np.sum(x_values[x_indicator], keepdims=True)/2.
    y = np.cumsum(x_values*x_indicator) * x_indicator / np.cumsum(x_indicator)
    y = np.nan_to_num(y)
    return x, y

def adding_data(T=3, batch_size=64, epoch_len=300):
    x = np.zeros((epoch_len, batch_size, T, 2))
    y = np.zeros((epoch_len, batch_size,T))
    for i in range(epoch_len):
        for b in range(batch_size):
            data = _adding_data(T)
            x[i][b] = data[0]
            y[i][b] = data[1]
    
    x = np.swapaxes(x,1,2).astype('float32')
    y = np.swapaxes(y,1,2).astype('float32')

    return x, y

if __name__ == "__main__":

    x,y = adding_data(T=10)

    print(x[0,:,0])
    print(y[0,:,0])

    print(x.shape, y.shape)

