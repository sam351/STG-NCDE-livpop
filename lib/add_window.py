import numpy as np

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def Add_Window_Horizon_Weekly(data, weeks_in=3, weeks_out=1, day_size=24):
    '''
    Generate Window Dataset Array by Week from Hourly/Daily Data
    :param data: shape [B, ...]
    :param weeks_in: number of weeks in the input sequence
    :param weeks_out: number of weeks in the output sequence
    :param day_size: number of data each day - when data is hourly data, day_size=24
    :return: X is [iter_num, window_in, ...], Y is [iter_num, window_out, ...]
    '''
    X = []  # input sequences
    Y = []  # output sequences
    iter_num = len(data) - day_size*7*(weeks_in + weeks_out -1) - day_size + 1
    for index in range(iter_num):
        X.append(np.concatenate([
            data[index+day_size*i:index+day_size*(i+1)] for i in range(0, 7*weeks_in, 7)
        ]))
        index = index+day_size*7*weeks_in
        Y.append(np.concatenate([
            data[index+day_size*i:index+day_size*(i+1)] for i in range(0, 7*weeks_out, 7)
        ]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


if __name__ == '__main__':
    from load_dataset import load_st_dataset
    import time
    st_time = time.time()
    
    dataset = 'LIVPOP'
    data = load_st_dataset(dataset)    
    print('>>> before shape :', data.shape)
    X, Y = Add_Window_Horizon_Weekly(data)
    print('>>> after shape :', X.shape, Y.shape)
    print(f'>>> Time Elapsed : {time.time()-st_time:.1f} s.')
