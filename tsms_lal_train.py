import pytorch_lightning as pl
import torch
import random
from typing import List, Dict
from termcolor import colored
import os
from helpers.monitoring import bin_metric_score_output
from db_loader.mongo_clicks import MongoClickLoader
from db_loader.clickhouse_raw_taxonomy import ClickhouseLoader as ClickhouseLoaderRaw
from helpers.data_helpers import merged_ind_slice_data
import collections
from configs.inds import industry_list # TODO move list to config.yml
from tqdm import tqdm
from helpers.data_helpers import build_dataset, build_loader
import pandas as pd
from helpers.const_parse import get_config
import numpy as np
import math
from sklearn.utils.class_weight import compute_class_weight
from vectorizer_cls import Vectorizer
from sklearn.preprocessing import StandardScaler
from models.lal_net_cls import LalNetClsCE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from models.autoenc import RawEmbAutoEncoder, IndustryEmbAutoEncoder
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import json
import time
from collections import Counter
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib

import warnings
warnings.filterwarnings('ignore')

def scale_data(data: pd.DataFrame, feature_columns: List[str], target_columns: List[str],
               industry_scaler_path: str, update_scaler=True,) -> pd.DataFrame:
    """
    scaling data by standard scaler[default]
    """
    if (industry_scaler_path is None) or (not os.path.exists(industry_scaler_path)):
        industry_scaler = StandardScaler()
    else:
        industry_scaler = joblib.load(industry_scaler_path)

    scaled_feature_dataframe = pd.DataFrame(industry_scaler.fit_transform(data[feature_columns]),
                            columns=feature_columns)
    targets = data.reset_index(drop=True)[target_columns]
    # if update_scaler:
    #     joblib.dump(industry_scaler, industry_scaler_path, compress=True)
    scaled_feature_dataframe[target_columns] = targets
    return  scaled_feature_dataframe


def merge_features_and_target_data(features: pd.DataFrame, target_data: pd.DataFrame, join_type='inner',
                                   features_index_name='subs_key', target_data_index_name='ctn') -> pd.DataFrame:

    """ merge features preprocessed taxonomy data and mongo click pre-target data for further training model """

    return pd.merge(
        features, target_data, how=join_type,
        left_on=features_index_name, right_on=target_data_index_name)

class TSMS_LaL:

    def __init__(self,
                 industry_list: list,
                 config_path: str='configs/industry_vectorizer_train_config.yml',
                 config_path_ae: str='configs/train_autoencoder_config.yml',
                 scalers_folder_path: str='tsms_bin_data/industry_scalers',
                 merged_data: pd.DataFrame=None,
                 actual_ctns_train: list=None,
                 class_weight_tensors_path='tsms_bin_data/weight_tensors.json',
                 accelerator='gpu',
                 devices_num=1):
        super(TSMS_LaL, self).__init__()

        self.industry_list = industry_list
        #device
        self.accelerator = accelerator
        self.devices_num = devices_num
        # config path
        self.config_path = config_path
        self.config_path_ae = config_path_ae
        self.class_weights_path = class_weight_tensors_path
        # scalers
        self.scalers_folder_path = scalers_folder_path

        # preloaded merged data
        self.merged_data = merged_data

        # actual ctns
        if actual_ctns_train is None:
            self.actual_ctns_train = list()
        else:
            self.actual_ctns_train = actual_ctns_train


        # dicts
        if config_path is not None:
            self.config_consts = get_config(config_path) # load from config
        else:
            raise (colored('No Config Error', 'red'))

        if config_path is not None:
            self.config_consts_ae = get_config(config_path_ae)  # load from config
        else:
            raise (colored('No Config AE Error', 'red'))

        self.industry_class_weight_tensor = dict()  # industry imbalance class weights

    def train_industry_classifier(self, industry: str, tax_features: pd.DataFrame,
                                  click_target_dataframe: pd.DataFrame):

        """ train lal classifier by current industry """

        print(colored(f'started with industry: {industry}', 'cyan'))
        industry_path_folder = f"checkpoints/{industry.replace('/', ' и ')}"
        industry_scaler_folder = f"tsms_bin_data/industry_scalers/{industry.replace('/', ' и ')}"

        # mkdir for further saving model checkpoints
        if not os.path.exists(f'{industry_path_folder}'):
            os.makedirs(industry_path_folder)
        # mkdir for industry scalers dumping
        if not os.path.exists(industry_scaler_folder):
            os.makedirs(industry_scaler_folder)


        dataset_ind, feature_columns, target_columns = self.prepare_merged_data_for_classifier_training(
            features=tax_features, target_data=click_target_dataframe, industry=industry,
            join_type='inner', features_index_name='subs_key',
            target_data_index_name='ctn',)
        print(colored(f'Training dataset ready', 'green'))

        # if self.config_consts['IS_SPLIT_DATA_TEST_TRAIN']:
        # split data
        train_data, test_data = train_test_split(dataset_ind, test_size=self.config_consts['TEST_SIZE'])
        train_features, test_features = train_data[feature_columns], test_data[feature_columns]
        train_target, test_target = train_data.target, test_data.target #train_data[target_columns], test_data[target_columns]

        input_shape = train_features.shape[1]

        # neg_size_test = test_target[test_target == 0].shape[0]
        # pos_size_test = test_target[test_target == 1].shape[0]

        # define imbalance classes weights
        w_classes = np.unique(train_target)
        w_weights = compute_class_weight(class_weight='balanced', classes=w_classes, y=train_target)
        w_class_weights = dict(zip(w_classes, w_weights))
        print(w_weights)
        WEIGHTS_TENSOR = torch.Tensor([int(math.ceil(w_class_weights[0])),
                                       int(w_class_weights[1] * 1.28)])
        print(colored(f'WEIGHT TENSOR {WEIGHTS_TENSOR}', 'cyan'))
        self.industry_class_weight_tensor[f"{industry.replace('/', ' и ')}"] = WEIGHTS_TENSOR.tolist()

        train_loader = build_loader(
            features=train_features, targets=train_target,
            shuffle=self.config_consts['SHUFFLE_DL'], batch_size=self.config_consts['BATCH_SIZE'])
        valid_loader = build_loader(
            features=test_features, targets=test_target,
            shuffle=self.config_consts['SHUFFLE_DL'], batch_size=self.config_consts['BATCH_SIZE'])
        # else: dataloader = build_loader(features=dataset[feature_cols_name], targets=dataset['target'])
        print(colored('Prepared data loaders | OK', 'green'))
        # CHECKPOINT_ROOT_DIR = f"checkpoints/{industry.replace('/', ' и ')}/"
        CHECKPOINT_ROOT_DIR = f"{self.config_consts['CHECKPOINT_ROOT_DIR']}/{industry.replace('/', ' и ')}/"



        # model init
        lal_net_ind_cls, early_stop_callback_ind, model_checkpointer_ind = self.init_industry_classifier(
            input_shape=input_shape, weight_tensor=WEIGHTS_TENSOR)
        print(colored('init model | OK', 'green'))

        # pl trainer init
        trainer = pl.Trainer(
            auto_lr_find=self.config_consts['AUTO_LR_FIND'], benchmark=self.config_consts['BENCHMARK'],
            deterministic=self.config_consts['DETERMINISTIC'], callbacks=[
                early_stop_callback_ind, model_checkpointer_ind],
            log_every_n_steps=self.config_consts['LOG_EVERY_N_STEPS'],
            default_root_dir=CHECKPOINT_ROOT_DIR, max_epochs=self.config_consts['MAX_EPOCHS'],
            accelerator=self.accelerator, devices=self.devices_num)

        print(colored(f'Started training with industry: {industry} | OK', 'green'))
        trainer.fit(lal_net_ind_cls, train_loader, valid_loader)

        if self.config_consts['CHECK_SCORES']:
            try:
                # neg_pred = test_target[test_target == 0].shape[0]
                # pos_pred = test_target[test_target == 0].shape[0]

                lal_net_ind_cls.eval()
                predict = lal_net_ind_cls(torch.from_numpy(test_features.astype('float32').values))
                predict_data = pd.Series(torch.argmax(predict, dim=1).numpy()).astype('float')

                print('===' * 25)
                print(f'predict positive shape: {predict_data[predict_data == 1].shape[0]}')
                print(f'predict negative shape: {predict_data[predict_data == 0].shape[0]}')

                print(f'ground truth positive shape: {test_target[test_target == 1].shape}')
                print(f'ground truth negative shape: {test_target[test_target == 0].shape}')
                print('===' * 25)
                bin_metric_score_output(test_target, predict_data)


                print(f'With industry : {industry}')
                #bin_metric_score_output(test_target, predict_data)
                print('===' * 30)
                train_predict = lal_net_ind_cls(torch.from_numpy(train_features.astype('float32').values))
                train_predict_data = pd.Series(torch.argmax(train_predict, dim=1).numpy()).astype('float')
                bin_metric_score_output(train_target, train_predict_data)
            except Exception as exc:
                print(f'corrupted with {exc} | industry name : {industry}')

        del dataset_ind
        del train_data, test_data, train_target, test_target
        del train_features, test_features

    def init_industry_classifier(self, input_shape: int, weight_tensor: torch.Tensor):
        """
        init industry classifier model and options for training
        return model, early_stopping, model_checkpointer
        """
        early_stop_callback = EarlyStopping(
            monitor=self.config_consts['MONITOR_LOSS'], min_delta=self.config_consts['MIN_DELTA'],
            patience=self.config_consts['EARLY_STOP_PATIENCE'], verbose=self.config_consts['PL_TRAINER_VERBOSE'],
            mode=self.config_consts['PL_TRAINER_MODE'])
        model_checkpointer = ModelCheckpoint(save_last=self.config_consts['SAVE_LAST'])
        lal_net_classifier = LalNetClsCE(input_shape=input_shape, view=False, weights_tensor=weight_tensor)
        lal_net_classifier.train()
        return lal_net_classifier, early_stop_callback, model_checkpointer

    def init_compress_autoencoder_tsms(self, input_shape=2144, hidden_shape=64):
        """
        init autoencoder model [compress] tsms
        """
        early_stop_callback_ae = EarlyStopping(
            monitor=self.config_consts_ae['MONITOR_LOSS'],
            min_delta=self.config_consts_ae['MIN_DELTA'],
            patience=self.config_consts_ae['EARLY_STOP_PATIENCE'],
            verbose=self.config_consts_ae['PL_TRAINER_VERBOSE'],
            mode=self.config_consts_ae['PL_TRAINER_MODE'])
        model_checkpointer_ea = ModelCheckpoint(save_last=self.config_consts['SAVE_LAST'])
        industry_ae = IndustryEmbAutoEncoder(encoder_input_size=input_shape,
                                        encoder_hidden_size=hidden_shape)
        industry_ae.train()
        return industry_ae, early_stop_callback_ae, model_checkpointer_ea

    def prepare_compress_train_data(self, tax_feature=None, #actual_ctns_csv_path: str=None,
                                    load_tax_features_path='tsms_data/taxonomy_preprocessed_dataframe.csv',
                                    tax_shape=323):
        """
        prepare train data for compress autoencoder training
        """
        # if actual_ctns_csv_path is not None:
        #     actual_ctns = list(pd.read_csv(actual_ctns_csv_path)[self.config_consts['saved_ctns_column_name']])
        # else:
        #     raise Exception(colored('actual ctns data error', 'red'))

        standard_scaler_ae = StandardScaler()
        if tax_feature is not None:
            tax_feature.drop('subs_key', axis=1, inplace=True)
            print(f'tax feature columns count {len(tax_feature.columns)}')
            assert len(tax_feature.columns) == tax_shape, 'useless columns error'

            features_column_name = list(tax_feature.reset_index(drop=True).columns)
            features_ae_train = pd.DataFrame(standard_scaler_ae.fit_transform(tax_feature),
                                    columns=features_column_name)
            targets_tax = pd.Series(list(features_ae_train.index))

        elif load_tax_features_path is not None:
            features_ae_train = pd.read_csv(load_tax_features_path)
            features_ae_train.reset_index(drop=True, inplace=True)
            features_ae_train.drop('subs_key', axis=1, inplace=True)
            print(f'features shape : {features_ae_train.shape}')
            features_column_name = list(features_ae_train.columns)
            features_ae_train = pd.DataFrame(standard_scaler_ae.fit_transform(features_ae_train),
                                    columns=features_column_name)
            joblib.dump(standart_scaler, 'tsms_bin_data/scaler/std_scaler_ae.bin', compress=True)
            targets_tax = pd.Series(list(features_ae_train.index))

            split_test_shape = math.ceil(features_ae_train.shape[0] * self.config_consts_ae['TEST_SIZE'])
            split_train_shape = features_ae_train.shape[0] - split_test_shape
            print(f'sizes of train: {split_train_shape} and test: {split_test_shape}')
        else:
            raise Exception(colored('Taxonomy data error', 'red'))

        return build_loader(features=features_ae_train, targets=targets_tax,
                                         batch_size=self.config_consts_ae['LOCAL_BATCH_SIZE'],
                                         shuffle=True, num_workers=self.config_consts_ae['NUM_WORKERS'])


    def prepare_merged_data_for_classifier_training(self, features, target_data, industry: str,
                    join_type='inner', features_index_name='subs_key', target_data_index_name='ctn', ):
        """
        preparing merged features data and pre-target data
        and build dataframe for training current industry classifier

        merged_data:  joined features and pre-target data
        industry: current industry name
        MAX_ADS_RATE_NEGATIVE: upper bound on click frequency (interest industry) for the negative class
        MIN_ADS_RATE_POSITIVE: lower bound on click frequency (industry interest) for the positive class
        batch_size_divisor: divisor value for batch size
        """

        merged_data = merge_features_and_target_data(features=features, target_data=target_data,
                                           join_type=join_type, features_index_name=features_index_name, target_data_index_name=target_data_index_name)

        if 'subs_key' in merged_data.columns:
            merged_data.drop('subs_key', axis=1, inplace=True)

        # provided that they were shown ads on the current industry (ads rate > 0)
        negative_interest_ctns = list(merged_data[(merged_data[f'ad_rate_{industry}'] > 0) & (
                merged_data[f'click_rate_{industry}'] < self.config_consts['MAX_ADS_RATE_NEGATIVE'])].ctn)
        positive_interest_ctns = list(merged_data[(merged_data[f'ad_rate_{industry}'] > 0) & (
                merged_data[f'click_rate_{industry}'] >= self.config_consts['MIN_ADS_RATE_POSITIVE'])].ctn)
        print(colored(f'negative interest ctns count: {len(negative_interest_ctns)}', 'cyan'))
        print(colored(f'positive interest ctns count: {len(positive_interest_ctns)}', 'cyan'))

        assert len(set(positive_interest_ctns).intersection(set(negative_interest_ctns))) == 0, \
            'positive and negative ctns intersection error'

        # build dataframe from defined positive and negative ctns
        negative_interest_df = merged_data[merged_data.ctn.isin(negative_interest_ctns)].reset_index(drop=True)
        positive_interest_df = merged_data[merged_data.ctn.isin(positive_interest_ctns)].reset_index(drop=True)

        # drop useless columns from interests dataframes
        negative_interest_df.drop(
            [f'ad_rate_{industry}', f'click_rate_{industry}'],
            axis=1, inplace=True)
        positive_interest_df.drop(
            [f'ad_rate_{industry}', f'click_rate_{industry}'],
            axis=1, inplace=True)

        # add binary target ([0, 1])
        negative_interest_df['target'] = 0
        positive_interest_df['target'] = 1

        concat_interest_dataframes = pd.concat((negative_interest_df, positive_interest_df), axis=0)
        print(colored(f'concat interest data size : {concat_interest_dataframes.shape}', 'cyan'))

        BATCH_SIZE = math.ceil(concat_interest_dataframes.shape[0] / self.config_consts['batch_size_divisor'])
        print(colored(f'BATCH SIZE value: {BATCH_SIZE}', 'cyan'))

        test_df_size = (concat_interest_dataframes.shape[0] * self.config_consts['TEST_SIZE'])

        # increasing the size of the dataframe with the same data,
        # provided that the batch size is less than the size of the test sample for calculating metrics
        if test_df_size < BATCH_SIZE:
            factor = math.ceil(BATCH_SIZE / test_df_size) + 1
            concat_interest_dataframes = pd.concat(([concat_interest_dataframes]*factor))
            print(f'increased concat interest data size : {concat_interest_dataframes.shape}')

        # delete useless variables
        del negative_interest_df
        del positive_interest_df
        del merged_data

        feature_columns = list(concat_interest_dataframes.reset_index(drop=True).drop(['ctn', 'target'], axis=1).columns)
        target_columns = ['target']
        # industry_scaler_path = f"tsms_bin_data/industry_scalers/{industry.replace('/', ' и ')}/std_scaler.bin"
        industry_scaler_path = 'raw_bin_data/scaler/std_scaler_ae.bin'

        dataset = scale_data(data=concat_interest_dataframes, feature_columns=feature_columns, target_columns=target_columns,
                             industry_scaler_path=industry_scaler_path, update_scaler=True)
        print(colored('Data scaled | OK', 'green'))
        return dataset, feature_columns, target_columns

    def save_imbalance_class_weights(self, path_imb_class_tensor_weights: str='tsms_bin_data/weight_tensors.json'):
        with open(path_imb_class_tensor_weights,  'w') as wtd:
            json.dump(self.industry_class_weight_tensor, wtd)


def define_actual_ctns(agg_click_and_ads_rate_data, industry_list: list,
                       actual_ctns_csv_save_path: str=None):
    """
    define all ctns which will participate in training
    """
    actual_ctns_train = list()
    ctns_ind_dict = collections.OrderedDict()
    for industry_name in industry_list:
        current_ind_ctns = list(
            agg_click_and_ads_rate_data[
                agg_click_and_ads_rate_data[
                    f'ad_rate_{industry_name}'] > 0].ctn.unique())
        ctns_ind_dict[industry_name] = current_ind_ctns
        print(colored(f'current industry : {industry_name} || ctns count : {len(current_ind_ctns)}', 'cyan'))

    for ind in list(ctns_ind_dict.keys()):
        actual_ctns_train.extend(ctns_ind_dict[ind])
    actual_ctns_train = list(set(actual_ctns_train))
    print(colored(f'Actual for training ctns count: {len(actual_ctns_train)}', 'cyan'))
    if actual_ctns_csv_save_path is not None:
        pd.DataFrame(self.actual_ctns_train, columns=[self.config_consts['saved_ctns_column_name']]).to_csv(actual_ctns_csv_save_path,
                                                                              index_label=False)
    return actual_ctns_train


def train_lal(industry_list: list, config_consts_path: str='configs/industry_vectorizer_train_config.yml',
            config_const_ae_path='configs/train_autoencoder_config.yml',
            print_config=True,
            merged_data: pd.DataFrame=None, actual_ctns_train: list=None,
            # class_weight_tensors_path='tsms_bin_data/weight_tensors.json',
            vectorizer_train=True, compress_encoder_train=True, tax_shape=323,
            lal_vec_output_tensor_shape: int = 32, ae_encode_shape: int = 64,
            pytorch_tensors_save_path='vecs_users2144.pt',
            update_json_weights_tensor=False, accelerator='gpu', devices_num=1):

    """
        train lal
        Choose what to train [industry classifiers, compress_autoencoder]
    """

    # config
    configs = get_config(config_consts_path)
    if print_config:
        for conf in configs:
            print(colored(f'{conf}: {configs[conf]}', 'yellow'))
    print(colored(f'Config loaded', 'green'))

    ae_input_shape = lal_vec_output_tensor_shape * len(industry_list)

    # init tsms lal class
    tsms_lal = TSMS_LaL(
        industry_list=industry_list,
        config_path=config_consts_path, config_path_ae=config_const_ae_path,
        scalers_folder_path=configs['INDUSTRY_SCALERS_FOLDER_PATH'],
        merged_data=merged_data,
        actual_ctns_train=actual_ctns_train,
        class_weight_tensors_path=configs['CLASS_WEIGHT_TENSOR_PATH'])
    print(colored('LaL Class initialized', 'green'))

    # init  db loaders #
    ########################################################################################
    # clickhouse_db_loader, mongo_db_loader = init_loaders(train_mode=True) # TODO wrap to function

    mongo_db_loader = MongoClickLoader(local=False, train_size=configs['TRAIN_SIZE'])
    mongo_db_loader.agg_click_save_to_csv('tsms_data/mongo/mongo_agg.csv')

    # mongo_db_loader = MongoClickLoader(local=True, saving_path='tsms_data/mongo/testing_tsms_lal_mongo_bank.csv')
    actual_train_ctns = define_actual_ctns(
        agg_click_and_ads_rate_data=mongo_db_loader.agg_click_and_ads_rates_dataframe,
        industry_list=industry_list)


    clickhouse_db_loader = ClickhouseLoaderRaw(local=False,
                                               train_ctns_size=configs['TRAIN_SIZE'],
                                               max_load_ctns_count=configs['MAX_LOAD_CTNS_COUNT'],
                                               clickhouse_load_batch_size=configs['CLICKHOUSE_LOAD_BATCH_SIZE'],
                                               actual_all_ctns=actual_train_ctns,
                                               mlbin_path=configs['MLBIN_PATH_FOLDER'],
                                               mlb_fitting=configs['MLBIN_FITING'])
    clickhouse_db_loader.preprocess_all_taxonomy(
        dropNA=True, drop_subs_keys_column=configs['DROP_SUBS_KEY_IN_PREPROCESS'])
    print(colored('DB Loaders initialized', 'green'))
    ########################################################################################

    # train vectorizer(industry classifiers)
    if vectorizer_train:
        print(colored('Start Training IND CLSs', 'green'))
        for industry_name in tsms_lal.industry_list:
            # get slice from mongo clicks data by industry
            industry_data_mongo, industry_ctns = mongo_db_loader.slice_ind_dataframe(industry_name=industry_name)
            tsms_lal.train_industry_classifier(industry=industry_name,
                                               tax_features=clickhouse_db_loader.features_taxonomy,
                                               click_target_dataframe=industry_data_mongo, )


    # generate for ae train
    configs_ae = get_config(config_const_ae_path)
    if print_config:
        for conf in configs_ae:
            print(colored(f'{conf}: {configs_ae[conf]}', 'yellow'))
    print(colored(f'Config loaded for autoencoder train', 'green'))
    if update_json_weights_tensor:
        tsms_lal.save_imbalance_class_weights(path_imb_class_tensor_weights=configs['WEIGHT_CLASS_TENSOR_JSON_PATH'])
# TODO json update
    del mongo_db_loader
    # init vectorizer
    lal_vectorizer = Vectorizer(industry_list=industry_list,
                                weights_json_path=configs['WEIGHT_CLASS_TENSOR_JSON_PATH'],
                                checkpoint_folder_path=configs['CHECKPOINT_ROOT_DIR'], input_shape=tax_shape)

    if configs_ae['VECTORIZE_AND_SAVING']:
        ae_train_loader = tsms_lal.prepare_compress_train_data(tax_feature=clickhouse_db_loader.features_taxonomy)

        user_vectors = list()
        c = 0

        for data in tqdm(ae_train_loader):
            c += 1
            features_batch, targets_batch = data
            print(features_batch.shape, targets_batch.shape)

            batch_embeddings = lal_vectorizer.vectorize_batch(features_batch)
            print(f'batch shape1 : {batch_embeddings.shape}')
            batch_embeddings = batch_embeddings.detach().view(batch_embeddings.shape[0],
                                                              batch_embeddings.shape[1] * batch_embeddings.shape[2])
            print(f'batch shape2 : {batch_embeddings.shape}')
            user_vectors.append(batch_embeddings)
            if c >= configs_ae['MAX_ITER_TRAIN_TENSORS_SAVING_COUNT']: # 1150
                break
        torch.save(torch.stack(user_vectors[:-1]).cpu(), configs_ae['TRAIN_TENSORS_PATH'])


    # init ae model and options
    if compress_encoder_train:
        industry_ae, early_stop_callback_ae, model_checkpointer_ea = tsms_lal.init_compress_autoencoder_tsms(
            input_shape=ae_input_shape, hidden_shape=ae_encode_shape)

        trainer = pl.Trainer(
            auto_lr_find=configs_ae['AUTO_LR_FIND'],
            benchmark=configs_ae['BENCHMARK'],
            deterministic=configs_ae['DETERMINISTIC'],
            callbacks=[early_stop_callback_ae, model_checkpointer_ea],
            log_every_n_steps=configs_ae['LOG_EVERY_N_STEPS'],
            default_root_dir=configs_ae['CHECKPOINT_ROOT_DIR'],
            max_epochs=configs_ae['MAX_EPOCHS'], accelerator=accelerator, devices=devices_num)

        user_vectors_torch = torch.load(configs_ae['TRAIN_TENSORS_PATH'])
        user_vec_shape = user_vectors_torch.shape
        print(f'torch tensor shape : {user_vec_shape}')

        user_vectors_torch = user_vectors_torch.view(user_vec_shape[0]*user_vec_shape[1],
                                                        user_vec_shape[2])
        user_vec_shape = user_vectors_torch.shape

        print(f'user vector torch shape: {user_vectors_torch.shape}')

        print(f'check : {user_vec_shape}')
        tmp_idxs = torch.ones(user_vec_shape[0])
        split_test_shape_ae = math.ceil(user_vec_shape[0] * configs_ae['TEST_SIZE'])
        split_train_shape_ae = user_vec_shape[0] - split_test_shape_ae

        print(f'split test size: {split_test_shape_ae}')
        print(f'split train size : {split_train_shape_ae}')
        print(f'temp idxs shape : {tmp_idxs.shape}')

        ae_dataset = build_dataset(features=user_vectors_torch, targets=tmp_idxs, tensor_type=True)
        train_dataset_ae, val_dataset_ae = torch.utils.data.random_split(ae_dataset, [split_train_shape_ae,
                                                                                      split_test_shape_ae])
        train_loader_ae = torch.utils.data.DataLoader(train_dataset_ae, batch_size=configs_ae['LOCAL_BATCH_SIZE'],
                                                      shuffle=True, num_workers=configs_ae['NUM_WORKERS'])
        valid_loader_ae = torch.utils.data.DataLoader(val_dataset_ae, batch_size=configs_ae['LOCAL_BATCH_SIZE'],
                                                      shuffle=False, num_workers=configs_ae['NUM_WORKERS'])
        print(colored(f'Started training AutoEncoder | OK', 'green'))
        #
        trainer.fit(industry_ae, train_loader_ae, valid_loader_ae)
    else:
        raise Exception(colored('Not specified what to train', 'red'))



if __name__ == '__main__':
    # industry_count = len(industry_list)
    # parts = 4
    # for i in range(parts):
    #     industry_part_list = industry_list[int(i*len(industry_list)/parts): int((i + 1)*(len(industry_list)/parts))]
    #     print(industry_part_list)
    start_time = time.time()
    train_lal(industry_list=industry_list,  vectorizer_train=True,
              compress_encoder_train=True, update_json_weights_tensor=False)
    print('====' * 10)
    print(f'all time: {round(time.time() - start_time, 3)}')






