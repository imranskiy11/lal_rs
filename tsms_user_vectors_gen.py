import math
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from vectorizer_cls import Vectorizer
from configs.inds import industry_list
from models.autoenc import IndustryEmbAutoEncoder
from termcolor import colored
from db_loader.clickhouse_raw_taxonomy import ClickhouseLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import struct
import os
import numpy as np
from helpers.const_parse import get_config
import torch
from datetime import datetime
import subprocess

def tsms_user_vector_generate(bin_file_path: str='tsms_bin_data/raw_lal_dim64.bin',
                              industry_list: list=industry_list, config_path: str='configs/tsms_user_generate_config.yml',
                              local=False, version_num: int=None, last_checkpoint_use=False,):

    config = get_config(config_path)
    checkpoint_root_dir_cls = config['INDUSTRY_VECTORIZER_CHECKPOINT_ROOT_DIR']
    checkpoint_root_dir_compress_ae = config['COMPRESS_AE_CHECKPOINT_ROOT_DIR']


    classifier_embeddings_shape = 32
    input_shape = 323 #len(industry_list)*classifier_embeddings_shape
    hidden_embeddings_shape = 64
    print(f'input shape: {input_shape}\nhidden shape: {hidden_embeddings_shape}\nclassifier model shape: {classifier_embeddings_shape}')

    # init vectorizer and build classifier model collection
    lal_vectorizer = Vectorizer(industry_list=industry_list, input_shape=input_shape,
                                weights_json_path=config['WEIGHTS_JSON_PATH'],
                                checkpoint_folder_path=checkpoint_root_dir_cls)

    if version_num is not None:
        last_version_num = version_num
    else: last_version_num = os.listdir(f'{checkpoint_root_dir_compress_ae}/lightning_logs')[-1]
    if last_checkpoint_use:
        ae_checkpoint_name = 'last.pt'
    else:  ae_checkpoint_name = os.listdir(
        f'{checkpoint_root_dir_compress_ae}/lightning_logs/{last_version_num}/checkpoints')[0]
    full_checkpoint_path = os.path.join(
        checkpoint_root_dir_compress_ae, 'lightning_logs', last_version_num, 'checkpoints', ae_checkpoint_name)

    # init compress autoencoder and loading checkpoint weights
    compress_autoencoder_tsms = IndustryEmbAutoEncoder().load_from_checkpoint(full_checkpoint_path)
    compress_autoencoder_tsms.eval()

    # print(compress_autoencoder_tsms)

    # # init clickhouse loader
    ch_loader = ClickhouseLoader(
        local=local,
        train_ctns_size=None,
        max_load_ctns_count=config['MAX_LOAD_CTNS_COUNT'],
        clickhouse_load_batch_size=config['CLICKHOUSE_LOAD_BATCH_SIZE'],
        train_mode=False)

    len_all_ctns = len(ch_loader.actual_all_ctns)
    print(f'CTNs count : {len_all_ctns}')

    scaler = joblib.load(config['SCALER_PATH'])

    with open(bin_file_path, 'wb') as f:
        for iteration_num in tqdm(range(math.ceil(len_all_ctns / config['CLICKHOUSE_LOAD_BATCH_SIZE']))):
            current_data, current_ctns = ch_loader.load_batch_tax(iteration_num, transform=True,
                                                                  batch_size=config['CLICKHOUSE_LOAD_BATCH_SIZE'])
            print(f'dataframe shape: {current_data.shape}')
            current_data.reset_index(drop=True, inplace=True)

            # transform to torch tensor
            current_data = torch.from_numpy(scaler.transform(current_data).astype('float32'))
            print(f'current data shape : {current_data.shape}')

            # vectorize
            current_tensors = lal_vectorizer.vectorize_batch(current_data)
            print(f'lal vectorized shape: {current_tensors.shape}')
            current_tensors = current_tensors.detach().view(current_tensors.shape[0],
                                                            current_tensors.shape[1] * current_tensors.shape[2])



            # autoencoder compression
            compressed_tensors_data = compress_autoencoder_tsms.encode(current_tensors)
            compressed_tensors_data = compressed_tensors_data.detach()

            print(f'compressed shape: {compressed_tensors_data.shape}')

            for ctn, users_vecs in tqdm(zip(current_ctns, compressed_tensors_data)):
                f.write(ctn.to_bytes(8, byteorder='little', signed=True))
                for user_vec in users_vecs:
                    f.write(struct.pack('<f', user_vec))
        f.close()







if __name__ == '__main__':
    today = datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
    industries = industry_list
    tsms_user_vector_generate(bin_file_path=f'tsms_bin_data/processed_tsms_{today}_dim_64.bin',
                              industry_list=industries,
                              config_path='configs/tsms_user_generate_config.yml')
    subprocess.call(f'mv tsms_bin_data/processed_tsms_{today}_dim_64.bin ready_to_copy_bin/tsms_{today}_dim_64.bin', shell=True)
