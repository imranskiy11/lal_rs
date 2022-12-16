from helpers.const_parse import get_config
from db_loader.clickhouse_raw_taxonomy import ClickhouseLoader
from models.autoenc import RawEmbAutoEncoder
from sklearn.preprocessing import StandardScaler
import torch
import pytorch_lightning as pl
import os
import struct
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from termcolor import colored
import math
from helpers.data_helpers import build_dataset, build_loader
from helpers.data_helpers import build_ae_dataset, build_ae_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from datetime import datetime
import subprocess
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    today = datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
    bin_file_path = f'raw_bin_data/processed_raw_{today}_dim_64.bin'
    # parse config
    config = get_config('configs/train_autoencoder_config_raw.yml')

    # load trained raw compress scaler
    raw_compress_scaler = joblib.load(config['SCALER_PATH'])

    #init and load compress autoencoder
    checkpoints_root_dir = config['CHECKPOINT_ROOT_DIR']
    checkpoint_folder = 'lightning_logs/' + os.listdir(os.path.join(checkpoints_root_dir, 'lightning_logs'))[-1] + '/checkpoints'

    checkpoint_file_name = os.listdir(os.path.join(checkpoints_root_dir, checkpoint_folder))[0]
    path_weights = '/'.join([checkpoints_root_dir, checkpoint_folder, checkpoint_file_name])
    print(f'compress model weigths file path: {path_weights}')
    compress_raw_ae = RawEmbAutoEncoder(encoder_input_size=config['INPUT_SHAPE'],
                      encoder_hidden_size=config['HIDDEN_EMB_SIZE']).load_from_checkpoint(f'{path_weights}')
    compress_raw_ae.eval()

    # init clickhouse loader
    ch_loader = ClickhouseLoader(
        local=False,
        train_ctns_size=config['TRAIN_SIZE'],
        max_load_ctns_count=config['MAX_LOAD_CTNS_COUNT'],
        clickhouse_load_batch_size=config['CLICKHOUSE_LOAD_BATCH_SIZE'],
        train_mode=False)

    ctns_count = len(ch_loader.actual_all_ctns)

    with open(bin_file_path, 'wb') as f:
        for iteration_num in tqdm(
                range(
                    math.ceil(ctns_count / config['CLICKHOUSE_LOAD_BATCH_SIZE']))):

            current_data, current_ctns = ch_loader.load_batch_tax(iteration_num, transform=True,
                                                                  batch_size=config['CLICKHOUSE_LOAD_BATCH_SIZE'])
            print(f'dataframe shape: {current_data.shape}')

            # transform to torch tensor
            current_data = torch.from_numpy(raw_compress_scaler.transform(current_data).astype('float32'))

            # autoencoder compression
            compressed_tensors_data = compress_raw_ae.encode(current_data)
            compressed_tensors_data = compressed_tensors_data.detach()

            print(f'compressed shape: {compressed_tensors_data.shape}')

            for ctn, users_vecs in tqdm(zip(current_ctns, compressed_tensors_data)):
                f.write(ctn.to_bytes(8, byteorder='little', signed=True))
                for user_vec in users_vecs:
                    f.write(struct.pack('<f', user_vec))
        f.close()


    subprocess.call(f'mv raw_bin_data/processed_raw_{today}_dim_64.bin ready_to_copy_bin/raw_{today}_dim_64.bin', shell=True)
