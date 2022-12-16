import torch
import pytorch_lightning as pl
import os
from torch.nn import functional as F
from models.lal_net_cls import LalNetClsCE
from configs.inds import industry_list
import collections
import json
from tqdm import tqdm
from models.autoenc import IndustryEmbAutoEncoder
import math
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from termcolor import colored
from helpers.data_helpers import build_dataset, build_loader


class Vectorizer:
    def __init__(self,
                 industry_list,
                 weights_json_path='tsms_bin_data/weight_tensors.json',
                 checkpoint_folder_path='tsms_bin_data/vectorize_cls_checkpoints',
                 input_shape=323,
                 version_num=None
                 ):
        self.version_num = version_num
        self.industry_list = industry_list
        self.input_shape = input_shape
        self.checkpoint_folder_path = checkpoint_folder_path

        self._build_industry_paths_dict()

        # loading weight tensors from json file
        with open(weights_json_path, 'r') as wt:
            self.weight_tensors = json.load(wt)

#        for i in self.weight_tensors:
#            print(i)
#            print('=='*30)

        self._build_cls_models_dict()

        print(f'clss count: {len(list(self.model_collection.keys()))}')

    def _build_industry_paths_dict(self):
        self.path_ind_dict = collections.OrderedDict()
        for industry_name in self.industry_list:
            replaced_ind_name = industry_name.replace('/', ' и ')

            if self.version_num is not None:
                last_version = f'version_{self.version_num}'
            else: last_version = os.listdir(f'{self.checkpoint_folder_path}/{replaced_ind_name}/lightning_logs')[-1]
            weight_file_name = os.listdir(f'{self.checkpoint_folder_path}/{replaced_ind_name}/lightning_logs/{last_version}/checkpoints')[0]
            weight_file_path = f'{self.checkpoint_folder_path}/{replaced_ind_name}/lightning_logs/{last_version}/checkpoints/{weight_file_name}'
            self.path_ind_dict[industry_name] = weight_file_path

    def _build_cls_models_dict(self):
        self.model_collection = collections.OrderedDict()
        for industry_name in tqdm(self.industry_list):
#            print(f'Models weights path: {self.path_ind_dict}')
            #if not isistance(self.weight_tensors, torch.Tensor):
            current_industry_model = LalNetClsCE.load_from_checkpoint(self.path_ind_dict[f'{industry_name}'],
                                input_shape=self.input_shape,
                                view=False,
                                weights_tensor=torch.Tensor(self.weight_tensors[f'{industry_name}'.replace('/', ' и ')])).eval()
            self.model_collection[f'{industry_name}'] = current_industry_model


    def vectorize_batch(self, batch_data, stack_dim=1):
        vectorize_chuck_list = list()
        for industry_name in self.industry_list:
            vectorize_chuck_list.append(self.model_collection[industry_name].vectorize(batch_data))
        return torch.stack(vectorize_chuck_list, dim=stack_dim)





if __name__ == '__main__':
    print('vectorizer imported')
