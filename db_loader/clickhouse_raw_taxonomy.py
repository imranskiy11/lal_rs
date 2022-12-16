from clickhouse_driver import connect as clickhouse_connect
import os
from tqdm import tqdm
from termcolor import colored
from .connections import clickhouse_connection_string, taxonomy_table_name
from sklearn.preprocessing import StandardScaler
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from configs.actual_columns_raw import actual_columns
import pandas as pd
import json
import itertools
from .connections import clickhouse_connection_string, taxonomy_table_name
from clickhouse_driver import connect as clickhouse_connect
from .preprocess_helpers import is_married, row_to_list, employ_replace_by_maping, remove_space_cls
from tqdm.rich import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from termcolor import colored
import math
import os
import pickle
from .preprocess_helpers import get_columns_names_with_NA_values, get_distrib, get_fill_list, list_to_string
import random
import warnings

warnings.filterwarnings('ignore')


class ClickhouseLoader:
    def __init__(self,
                 local=True,
                 save_train_loaded_data=False,
                 batch_mode=False,
                 columns_conf_json_path='configs/col_json_data.json',
                 actual_columns=actual_columns,
                 connection_string=None,
                 actual_all_ctns=None,
                 train_ctns_size=500000,
                 max_load_ctns_count=50000000,
                 clickhouse_load_batch_size=100000,
                 loaded_data_saving_path_csv=None,
                 prepared_csv_path='data/clickhouse_actual.csv',
                 saving_taxonomy_dataframe_path='data/taxonomy_dataframe.csv',
                 mlbin_path='tsms_bin_data/mlb_data',
                 mlb_fitting=False, train_mode=True,):
        super(ClickhouseLoader, self).__init__()

        self.actual_tax_columns = actual_columns
        self.train_ctns_size = train_ctns_size

        self.local = local
        self.save_train_loaded_data = save_train_loaded_data
        self.prepared_csv_path = prepared_csv_path
        self.saving_path = saving_taxonomy_dataframe_path
        self.loaded_data_saving_path_csv = loaded_data_saving_path_csv
        self.mlbin_path = mlbin_path
        self.mlb_fit = mlb_fitting
        self.batch_mode = batch_mode
        self.train_mode = train_mode

        self.actual_all_ctns = actual_all_ctns

        if os.path.exists(self.saving_path):
            self.local_load = True
        else:
            self.local_load = False

        self.max_load_ctns_count = max_load_ctns_count
        self.clickhouse_load_batch_size = clickhouse_load_batch_size

        ################ define columns(features) from taxonomy #############
        self.columns_conf_json_path = columns_conf_json_path
        self.columns_list = self._build_columns_list()
        self.main_columns = list(self.columns_list['main_features'].keys())
        if 'subs_key' not in self.main_columns:
            self.main_columns.append('subs_key')

        self.conc_features_list = list(itertools.chain(
            self.main_columns,
            list(self.columns_list['interest_features'].keys()),
            list(self.columns_list['additional_features'].keys()),
            list(self.columns_list['bank_and_finance_features'].keys())
        ))
        # print(f'needed columns count: {len(self.conc_features_list)}')
        ######################################################################

        if not self.local:
            # build connection to clickhouse taxonomy data
            if connection_string is None:
                self.clickhouse_connection_cursor = clickhouse_connect(clickhouse_connection_string).cursor()
            else:
                self.clickhouse_connection_cursor = clickhouse_connect(connection_string).cursor()

            self.define_all_columns_list()

            # all columns list from clickhouse db
            self.clickhouse_connection_cursor.execute(f'DESCRIBE TABLE {taxonomy_table_name}')
            all_columns_list = [col_name[0] for col_name in self.clickhouse_connection_cursor.fetchall()]

            if self.actual_all_ctns is None:  # TODO batch_size load test
                # if loaded all ctns data
                ctns_query = f'select top {self.max_load_ctns_count} subs_key from FocusTaxonomy_prod'
                self.clickhouse_connection_cursor.execute(ctns_query)
                self.all_ctns_out = self.clickhouse_connection_cursor.fetchall()
                self.all_ctns_out = [t[0] for t in self.all_ctns_out]
                self.all_ctns_out = list(set(self.all_ctns_out))
                self.actual_all_ctns = self.all_ctns_out
                print(f'all ctns count : {len(self.all_ctns_out)}')
                # raise Exception('Enter actual all ctns from mongo loader\nTODO about batch size and loading time testing')
            else:
                self.actual_all_ctns = actual_all_ctns

            # if not self.batch_mode:
            #     # load actual data for training
            #     if self.local_load:
            #         self.taxonomy_dataframe = pd.read_csv(self.saving_path)
            #     else:
            #         self.load_train_data()
            # else:
            #     pass
            # print(colored('loaded click data with actual ctns | OK', 'green'))
            if self.train_mode:
                self.load_train_data(ctns=self.actual_all_ctns)
            else:
                pass
        else:
            # test mode and load prepared csv data from clickhouse
            self.taxonomy_dataframe = pd.read_csv(self.prepared_csv_path)
            print(colored('loaded prepared csv-file | OK', 'green'))

    def define_all_columns_list(self):
        self.clickhouse_connection_cursor.execute(f'DESCRIBE TABLE {taxonomy_table_name}')
        self.all_columns_list = [col_name[0] for col_name in self.clickhouse_connection_cursor.fetchall()]


    def set_columns_json_file(self, path):
        self.columns_conf_json_path = path

    def _build_columns_list(self) -> dict:
        with open(self.columns_conf_json_path, 'r') as json_file:
            columns_dict = json.load(json_file)
        return columns_dict



    def load_train_data(self, ctns=None):
        """ load tax data for training """
#        if ctns is None:
        if len(self.actual_all_ctns) > self.train_ctns_size:
            ctns = random.sample(self.actual_all_ctns, k=self.train_ctns_size)

        tables_tax = ', '.join(self.actual_tax_columns)

        clickhouse_dataframe_list = list()

        for i in tqdm(range(math.ceil(len(ctns) / self.clickhouse_load_batch_size))):
            current_subs_key = ctns[
                               i * self.clickhouse_load_batch_size:(i + 1) * self.clickhouse_load_batch_size]
            query = f'select {tables_tax} from FocusTaxonomy_prod where subs_key in {current_subs_key}'
            self.clickhouse_connection_cursor.execute(query)
            output = self.clickhouse_connection_cursor.fetchall()


            current_click_df = pd.DataFrame(output, columns=self.actual_tax_columns)
            print(f'current iter data shape: {current_click_df.shape}')
            clickhouse_dataframe_list.append(current_click_df)
            self.taxonomy_dataframe = pd.concat(clickhouse_dataframe_list)

        del clickhouse_dataframe_list
        del current_click_df
        print(f'taxonomy features size : {self.taxonomy_dataframe.shape}')
        if self.save_train_loaded_data:
            if self.loaded_train_data_saving_path_csv is not None:
                if not os.path.exists(self.loaded_train_data_saving_path_csv):
                    self.taxonomy_dataframe.to_csv(self.loaded_data_saving_path_csv, index_label=False)

    def load_batch_tax(self, iteration_num, transform=True,
                       tax_columns:list =None, batch_size=10000):
        """ load batch taxonomy data for generate user embeddings further"""
        if tax_columns is None:
            tables_tax = ', '.join(self.actual_tax_columns)
        else: tables_tax = ', '.join(tax_columns)
        current_ctns = self.actual_all_ctns[iteration_num * batch_size:(iteration_num + 1) * batch_size]
        query_tax_batch_load = f'select {tables_tax} from FocusTaxonomy_prod where subs_key in {current_ctns}'
        self.clickhouse_connection_cursor.execute(query_tax_batch_load)
        self.taxonomy_dataframe = pd.DataFrame(self.clickhouse_connection_cursor.fetchall(),
                                               columns=self.actual_tax_columns)

        if transform:
            self.preprocess_all_taxonomy(dropNA=False)
        return self.features_taxonomy, current_ctns



    def taxonomy_dataframe_setter(self, with_null=False):
        # TODO
        pass

    def preprocess_all_taxonomy(self, dropNA=True,
                                object_cols_to_preprocess=['os', 'age_cat',
                                                          'family_status',
                                                          'employment_type'],
                                drop_subs_keys_column=True):
        """
        prepare and preprocess taxonomy dataframe
        """
        # object_useful_columns = ['family_status', 'employment_type', 'mcc_category']
        self.features_taxonomy = self.taxonomy_dataframe[self.actual_tax_columns]
        self.features_taxonomy.drop(  # remove birth columns
            ['birth_day', 'birth_month', 'birth_year'], axis=1, inplace=True)

        # columns where any NaN values
        if not dropNA:
            with_null_cols = get_columns_names_with_NA_values(self.features_taxonomy)
            for col_name in tqdm(with_null_cols):
                self.features_taxonomy.reset_index(drop=True, inplace=True)
                null_idxs = list(self.features_taxonomy[self.features_taxonomy[col_name].isnull()].index)
                fill_list = get_fill_list(self.features_taxonomy, col_name)
                assert len(fill_list) == len(null_idxs), 'length error'
                self.features_taxonomy[col_name][self.features_taxonomy.index.isin(null_idxs)] = fill_list
            print(f'NA df shape : {self.features_taxonomy[self.features_taxonomy.isnull()].shape}')
            assert len(get_columns_names_with_NA_values(self.features_taxonomy)) == 0, 'nulls error'

        print(f'features taxonomy shape : {self.features_taxonomy.shape}')

        if dropNA:
            self.features_taxonomy.dropna(inplace=True)

        ############################# os #########################################
        if 'os' in object_cols_to_preprocess:
            self.features_taxonomy['os'][self.features_taxonomy['os'] == 2] = 0
            self.features_taxonomy['os'][self.features_taxonomy['os'] != 0] = 1
            self.features_taxonomy['os'] = self.features_taxonomy['os'].astype(int)
        ###########################################################################

        ############################# age_cat #####################################
        if 'age_cat' in object_cols_to_preprocess:
            self.features_taxonomy['age_cat'] = self.features_taxonomy['age_cat'].interpolate().apply(round)
            self.features_taxonomy['age_cat'] = self.features_taxonomy['age_cat'].astype(int)
            self.features_taxonomy = pd.concat(
                (self.features_taxonomy, pd.get_dummies(self.features_taxonomy.age_cat, prefix='age_cat')), axis=1)
            self.features_taxonomy.drop('age_cat', axis=1, inplace=True)
        ###########################################################################

        ############################# family status ##############################
        if 'family_status' in object_cols_to_preprocess:
            self.features_taxonomy['family_status'] = self.features_taxonomy['family_status'].apply(is_married)
            self.features_taxonomy['family_status'] = self.features_taxonomy['family_status'].interpolate().apply(round)
            self.features_taxonomy['family_status'] = self.features_taxonomy['family_status'].astype(int)
        ###########################################################################

        ############################# employment type #############################
        if 'employment_type' in object_cols_to_preprocess:
            self.features_taxonomy['employment_type'] = self.features_taxonomy['employment_type'].astype(str)
            self.features_taxonomy['employment_type'] = self.features_taxonomy['employment_type'].apply(
                employ_replace_by_maping)
        if self.mlb_fit:
            mlb_emp = MultiLabelBinarizer()
        else:
            with open('tsms_bin_data/mlb_data/multi_label_binarizer_emp.pkl', 'rb') as f:  #(f'{self.mlbin_path}/multi_label_binarizer_emp.pkl', "rb") as f:
                mlb_emp = pickle.load(f)
        category_list = list()
        for row in tqdm(self.features_taxonomy.employment_type):
            category_list.append(row_to_list(row))
        self.features_taxonomy['employed_type'] = category_list
        category_df = self.features_taxonomy[['subs_key', 'employed_type']]
        self.features_taxonomy.drop(['employed_type', 'employment_type'], axis=1, inplace=True)
        if self.mlb_fit:
            mlb_emp.fit(category_df['employed_type'])
        # print(f'MLB Classes : {mlb_emp.classes_}')
        self.features_taxonomy = pd.merge(
            self.features_taxonomy,
            pd.DataFrame(mlb_emp.transform(category_df['employed_type']),
                         columns=mlb_emp.classes_),
            left_index=True,
            right_index=True)
        del category_df, category_list
        self.features_taxonomy.drop('unknown_emp', axis=1, inplace=True)
        ###########################################################################


        # mcc_category
        if self.mlb_fit:
            mlb_mcc = MultiLabelBinarizer()
        else:
            with open('tsms_bin_data/mlb_data/multi_label_binarizer_mcc.pkl', 'rb') as f: #open(f'{self.mlbin_path}/multi_label_binarizer_mcc.pkl', "rb") as f:
                mlb_mcc = pickle.load(f)

        self.features_taxonomy.mcc_category = self.features_taxonomy.mcc_category.apply(
            list_to_string)  # TODO temporary solution
        category_list = list()
        for row in tqdm(self.features_taxonomy.mcc_category):
            category_list.append(row_to_list(row))

        self.features_taxonomy['mcc_cat'] = category_list
        category_df = self.features_taxonomy[['subs_key', 'mcc_cat']]
        self.features_taxonomy.drop(['mcc_cat', 'mcc_category'], axis=1, inplace=True)
        if self.mlb_fit:
            mlb_mcc.fit(category_df['mcc_cat'])
        # print(f'MLB Classes : {mlb_mcc.classes_}')
        mlb_array_mcc_cat = remove_space_cls(np.where(mlb_mcc.classes_ == '', 'unknown_mcc_cat', mlb_mcc.classes_))
        self.features_taxonomy = pd.merge(
            self.features_taxonomy,
            pd.DataFrame(mlb_mcc.transform(category_df['mcc_cat']),
                         columns=mlb_array_mcc_cat).add_prefix('mcc_cat_'),
            left_index=True,
            right_index=True)
        del category_df, category_list
        self.features_taxonomy.drop('mcc_cat_unknown_mcc_cat', axis=1, inplace=True)

        # drop useless columns
        for useless_column in ['car_interest_brand', 'debit_card_bank_name', 'credit_card_bank_name',
                               'car_owner_brand']:
            try:
                self.features_taxonomy.drop(useless_column, axis=1, inplace=True)
            except Exception as drop_error:
                # print(f"useless columns can't deleted with exception {drop_error}")
                pass
        print(f'Taxonomy features dataframe shape: {self.features_taxonomy.shape} | ')

        # drop rows with nan
        if dropNA:
            self.features_taxonomy.dropna(inplace=True)

        # drop duplicated
        self.features_taxonomy.drop_duplicates(subset='subs_key', inplace=True)

        # drop ctns column
        if drop_subs_keys_column:
            self.features_taxonomy.drop(
                'subs_key', axis=1, inplace=True)

        # drop old indexes
        self.features_taxonomy.reset_index(drop=True, inplace=True)


    @property
    def get_features_taxonomy_list(self):
        if bool(self.features_taxonomy):
            return list(self.features_taxonomy.columns)
        return None

    #

    def build_all_ctns(self):
        ctns_query = f'select top {self.max_load_ctns_count} subs_key from FocusTaxonomy_prod'
        self.all_ctns_out = self.clickhouse_connection.cursor.fetchall()
        self.all_ctns_out = [t[0] for t in self.all_ctns_out]
        self.all_ctns_out = list(set(self.all_ctns_out))

    def get_batch_tax(self, iteration_num, inner_batch_size=100000):
        # define ctns
        # if self.all_ctns_out is None:
        #     self.build_all_ctns() # self.all_ctns_out

        tables_tax = ', '.join(self.conc_features_list)
        current_ctns = self.all_ctns_out[iteration_num * inner_batch_size:(iteration_num + 1) * inner_batch_size]
        query = f'select {tables_tax} from FocusTaxonomy_prod where subs_key in {current_ctns}'
        self.clickhouse_connection_cursor.execute(query)
        output = self.clickhouse_connection_cursor.fetchall()
        self.taxonomy_dataframe = pd.DataFrame(output, columns=self.conc_features_list)
        self.preprocess_taxonomy(dropNA=False)
        return self.features_taxonomy

#
# class ClickhouseRawTax:
#     def __init__(self, columns_data_file_path, local=False,
#                  max_load_ctns_count=50000000,
#                  connection_string=None,
#                  clichhouse_load_batch_size=100000,
#                  mlbin_path=None, mlb_fitting=False,
#                  table_name=None, scaler_path=None,
#                  train_scaler_size=10000):
#
#         super(ClickhouseRawTax, self).__init__()
#
#         if not self.local:
#             # build connection to clickhouse taxonomy data
#             if connection_string is None:
#                 self.clickhouse_connection_cursor = clickhouse_connect(clickhouse_connection_string).cursor()
#             else:
#                 self.clickhouse_connection_cursor = clickhouse_connect(connection_string).cursor()
#
#         self.table_name = taxonomy_table_name
#
#         # execute all ctns query
#         self.clickhouse_connection_cursor.execute(
#             f'select top {max_load_ctns_count} subs_key from {self.table_name}')
#         self.all_ctns = [
#             cell[0] for cell in self.clickhouse_connection_cursor.fetchall()]
#
#         # assert isinstance(self.all_ctns, list), 'Type error'
#         self.train_scaler_sample_ctns = train_scaler_size
#
#         # init or load scaler
#         if scaler_path is None:
#             self.scaler = StandardScaler()
#         else: self.scaler = joblib.load(scaler_path)
#
#         # all columns list from clickhouse db
#         self.clickhouse_connection_cursor.execute(f'DESCRIBE TABLE {taxonomy_table_name}')
#         all_columns_list = [col_name[0] for col_name in self.clickhouse_connection_cursor.fetchall()]
#
#         self.conc_columns = None
#
#
#     def init_train_data(self, prepared_csv_path=None):
#         # if local load dataframe taxonomy
#         if prepared_csv_path is not None:
#             return pd.read_csv(prepared_csv_path)
#         print(self.table_name)
#         # TODO load data
#
#
#     def fill_data(self, cols):
#         # fill data by cols
#         pass
#
#     def to_binary_data(self, cols):
#         #change to binary data input columns
#         pass
#
#     def transform(self, cols):
#         # transform and feature engineering
#         pass


if __name__ == '__main__':
    print('init clickhouse loader class')




