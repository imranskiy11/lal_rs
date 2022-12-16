import pandas as pd
import torch

def merge_mongo_and_clickhouse_data(clickhouse_data, mongo_data, mongo_columns=['ctn', 'clicked']):
    return pd.merge(
        mongo_data[mongo_columns],
        clickhouse_data,
        how='inner',
        left_on='ctn',
        right_on='subs_key'
    )
    

def merged_ind_slice_data(taxonomy_data: pd.DataFrame, slice_mongo_data: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        taxonomy_data,
        slice_mongo_data,
        how='inner',
        left_on='ctn',
        right_on='ctn')


def build_dataset(features, targets, tensor_type=False):
    if not tensor_type:
        features = torch.from_numpy(features.astype('float32').values)
        targets = torch.from_numpy(targets.values.astype('float32'))
    return torch.utils.data.TensorDataset(features, targets)


def build_loader(features, targets, shuffle=True, batch_size=512, num_workers=0):
    dataset = build_dataset(features, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)


def build_ae_dataset(features, tensor_type=False):
    if not tensor_type:
        features = torch.from_numpy(features.astype('float32').values)
    return  torch.utils.data.TensorDataset(features)

def build_ae_dataloader(features, shuffle=True, batch_size=4096, num_workers=0):
    dataset = build_ae_dataset(features)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)
