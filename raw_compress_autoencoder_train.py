from helpers.const_parse import get_config
from db_loader.clickhouse_raw_taxonomy import ClickhouseLoader
from models.autoenc import RawEmbAutoEncoder
from sklearn.preprocessing import StandardScaler
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from termcolor import colored
import math
from helpers.data_helpers import build_dataset, build_loader
from helpers.data_helpers import build_ae_dataset, build_ae_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib

import warnings
warnings.filterwarnings('ignore')


def model_and_trainer(configs:dict, input_shape=323, accelerator='gpu', devices_num=1):
    """
    init autoencoder model [compress] raw lal and trainer
    """
    early_stop_callback_ae = EarlyStopping(
        monitor=configs['MONITOR_LOSS'],
        min_delta=configs['MIN_DELTA'],
        patience=configs['EARLY_STOP_PATIENCE'],
        verbose=configs['PL_TRAINER_VERBOSE'],
        mode=configs['PL_TRAINER_MODE'])
    model_checkpointer_ea = ModelCheckpoint(save_last=configs['SAVE_LAST'])

    compress_autoencoder = RawEmbAutoEncoder(
        encoder_input_size=input_shape,
        encoder_hidden_size=configs['HIDDEN_EMB_SIZE'])
    compress_autoencoder.train()

    raw_trainer = pl.Trainer(
        auto_lr_find=configs['AUTO_LR_FIND'],
        benchmark=configs['BENCHMARK'],
        deterministic=configs['DETERMINISTIC'],
        callbacks=[early_stop_callback_ae, model_checkpointer_ea],
        log_every_n_steps=configs['LOG_EVERY_N_STEPS'],
        default_root_dir=configs['CHECKPOINT_ROOT_DIR'],
        max_epochs=configs['MAX_EPOCHS'],
        accelerator=accelerator, devices=devices_num)

    return compress_autoencoder, raw_trainer


if __name__ == '__main__':
    # parse config
    config = get_config('configs/train_autoencoder_config_raw.yml')

    # init scaler
    scaler = StandardScaler()

    # init clickhouse loader
    ch_loader = ClickhouseLoader(
        local=False,
        train_ctns_size=config['TRAIN_SIZE'],
        max_load_ctns_count=config['MAX_LOAD_CTNS_COUNT'],
        clickhouse_load_batch_size=config['CLICKHOUSE_LOAD_BATCH_SIZE'])

    ch_loader.preprocess_all_taxonomy(dropNA=config['DROPNA'],
                                      drop_subs_keys_column=True)

    taxonomy_train_data_shape = ch_loader.features_taxonomy.shape

    # init encoder and trainer
    compress_ae, trainer = model_and_trainer(configs=config, input_shape=taxonomy_train_data_shape[-1])

    scaler.fit(ch_loader.features_taxonomy)
    joblib.dump(scaler, config['SCALER_PATH'], compress=True)

    if config['GENERATE_NEW_TRAIN_AE_DATA']:
        # create or update train data tensors
        train_tensors_data = torch.from_numpy(scaler.fit_transform(ch_loader.features_taxonomy).astype('float32'))
        torch.save(train_tensors_data.cpu(), config['TRAIN_DATA_TENSOR_PATH'])
    else:
        if not os.path.exists(config['TRAIN_DATA_TENSOR_PATH']):
            raise Exception(colored('Train tensors file {pt} not exists\nTry to generate new train tensor', 'red'))
        else:
            train_tensors_data = torch.load(config['TRAIN_DATA_TENSOR_PATH'])
    del ch_loader

    tmp_idxs = torch.ones(train_tensors_data.shape[0])
    split_test_shape_ae = math.ceil(train_tensors_data.shape[0] * config['TEST_SIZE'])
    split_train_shape_ae = train_tensors_data.shape[0] - split_test_shape_ae

    print(f'split test size: {split_test_shape_ae}')
    print(f'split train size : {split_train_shape_ae}')
    print(f'temp idxs shape : {tmp_idxs.shape}')

    ae_dataset = build_dataset(features=train_tensors_data, targets=tmp_idxs, tensor_type=True)
    train_dataset_ae, val_dataset_ae = torch.utils.data.random_split(
        ae_dataset, [split_train_shape_ae, split_test_shape_ae])
    del train_tensors_data
    train_loader_ae = torch.utils.data.DataLoader(train_dataset_ae, batch_size=config['LOCAL_BATCH_SIZE'],
                                                  shuffle=True, num_workers=config['NUM_WORKERS'])
    valid_loader_ae = torch.utils.data.DataLoader(val_dataset_ae, batch_size=config['LOCAL_BATCH_SIZE'],
                                                  shuffle=False, num_workers=config['NUM_WORKERS'])
    print(colored(f'Started training Raw Compress AutoEncoder | OK', 'green'))
    trainer.fit(compress_ae, train_loader_ae, valid_loader_ae)

    print(colored('Finish raw lal compress autoencoder train', 'green'))
