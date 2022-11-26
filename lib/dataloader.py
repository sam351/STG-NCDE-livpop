if __name__ == '__main__':
    import time
    st_time = time.time()

    import os
    import sys
    file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(file_dir)
    print(file_dir)


import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon, Add_Window_Horizon_Weekly
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import controldiffeq

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)
    # X = tuple(TensorFloat(x) for x in X)
    # Y = TensorFloat(Y)
    data = torch.utils.data.TensorDataset(*X, torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_dataloader_cde(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    # TODO: make argument for missing data
    if args.missing_test == True:
        generator = torch.Generator().manual_seed(56789)
        xs = np.concatenate([x_tra, x_val, x_test])
        for xi in xs:
            removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[:int(xs.shape[1] * args.missing_rate)].sort().values
            removed_points_node = torch.randperm(xs.shape[2], generator=generator)[:int(xs.shape[2] * args.missing_rate)].sort().values

            for seq in removed_points_seq:
                for node in removed_points_node:
                    xi[seq,node] = float('nan')
        x_tra = xs[:x_tra.shape[0],...] 
        x_val = xs[x_tra.shape[0]:x_tra.shape[0]+x_val.shape[0],...]
        x_test = xs[-x_test.shape[0]:,...] 
    
    #
    data_category = args.data_category
    if data_category == 'traffic':
        times = torch.linspace(0, 11, 12)
    elif data_category == 'token':
        times = torch.linspace(0, 6, 7)
    elif data_category == 'livpop':
        times = torch.linspace(0, x_tra.shape[1]-1, x_tra.shape[1])
    else:
        raise ValueError
    
    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

    ###
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1,2))
    # train_coeffs = tuple(coeff.transpose(1,2) for coeff in train_coeffs)
    # valid_coeffs = tuple(coeff.transpose(1,2) for coeff in valid_coeffs)
    # test_coeffs = tuple(coeff.transpose(1,2) for coeff in test_coeffs)

    #get dataloader
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler, times


def get_dataloader_cde_v2(args, normalizer = 'std', single=False, day_size=24, verbose=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    if verbose:
        print(f'\n>>> Load {args.dataset} Dataset shaped: {data.shape}')
    
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    if verbose:
        print(f'\n>>> After normalization shaped:', data.shape)
    
    #add time window
    data_category = args.data_category
    if data_category == 'livpop':
        x_data, y_data = Add_Window_Horizon_Weekly(data, args.lag, args.horizon, day_size)
    else:
        x_data, y_data = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    if verbose:
        print(f'\n>>> After window split:')
        print(x_data.shape, y_data.shape)
    
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        x_tra, x_val, x_test = split_data_by_days(x_data, args.val_ratio, args.test_ratio)
        y_tra, y_val, y_test = split_data_by_days(y_data, args.val_ratio, args.test_ratio)
    else:
        x_tra, x_val, x_test = split_data_by_ratio(x_data, args.val_ratio, args.test_ratio)
        y_tra, y_val, y_test = split_data_by_ratio(y_data, args.val_ratio, args.test_ratio)
    if verbose:
        print(f'\n>>> After split:')
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    
    #add row id
    if data_category == 'traffic':
        times = torch.linspace(0, 11, 12)
    elif data_category == 'token':
        times = torch.linspace(0, 6, 7)
    elif data_category == 'livpop':
        times = torch.linspace(0, x_tra.shape[1]-1, x_tra.shape[1])
    else:
        raise ValueError
    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

    #get coeffs
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1,2))

    #get dataloader
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler, times


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default='LIVPOP', type=str)
    # parser.add_argument('--num_nodes', default=1000, type=int)  # #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=3, type=int)
    parser.add_argument('--horizon', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--column_wise', default=False, type=eval)
    parser.add_argument('--missing_test', default=False, type=bool)
    parser.add_argument('--data_category', default='livpop', type=str)
    args = parser.parse_args()
    print(args)
    
    train_dataloader, val_dataloader, test_dataloader, scaler, times = get_dataloader_cde_v2(args, normalizer='std', single=False, verbose=True)
    print('\n>>> train_dataloader:', len(train_dataloader), train_dataloader.__class__)
    print('>>> val_dataloader:', len(val_dataloader), val_dataloader.__class__)
    print('>>> test_dataloader:', len(test_dataloader), test_dataloader.__class__)
    print('>>> scaler:', scaler.__class__)
    print('>>> times:', times)
    print(f'\n>>> Time Elapsed : {time.time()-st_time:.1f} s.')
