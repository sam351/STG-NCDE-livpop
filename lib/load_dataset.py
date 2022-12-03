import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        # data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD7M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD7L':
        data_path = os.path.join('../data/PEMS07L/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'Decentraland':
        data_path = os.path.join('../token_data/Decentraland_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    elif dataset == 'Bytom':
        data_path = os.path.join('../token_data/Bytom_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    elif dataset == 'LIVPOP':
        data_path = os.path.join('../data/LIVPOP/LIVPOP.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, living population data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print(f'Load {dataset} Dataset shaped: {data.shape}')
    print(f'min: {data.min():.4f}  &  max: {data.max():.4f}  &  median: {np.median(data):.4f}')
    print(f'mean: {data.mean()}')
    print(f'std: {data.std()}')
    print()
    return data


if __name__ == '__main__':
    import argparse
    
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LIVPOP', type=str)
    args = parser.parse_args()
    print(args)
    
    # test loading dataset
    data = load_st_dataset(args.dataset)
