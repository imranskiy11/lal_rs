from typing import Tuple
from pymongo import MongoClient
import pandas as pd
from tqdm.rich import tqdm
from .connections import mongo_connect_dict
from termcolor import colored
import os

class MongoClickLoader:
    def __init__(self,
                connect_params=None, train_size=None,
                saving_path='tsms_data/mongo/mongo_agg.csv',
                local=True):
        super(MongoClickLoader).__init__()
        
        self.saving_path_mongo_df = saving_path
        # MongoDB connection parameters
        if connect_params is None:
            self.connect_params = mongo_connect_dict
        else:
            self.connect_params = connect_params

        self.train_size = train_size

        self.local = local

        # build mongodb collection
        self.collection_clicks = self._build_mongo_collection()


        # aggregate clicks and rates from mongo dataframe building
        if self.local:
            if os.path.exists(self.saving_path_mongo_df):
                self.agg_click_and_ads_rates_dataframe = pd.read_csv(self.saving_path_mongo_df)
            else:
                raise Exception(colored(f'CSV file not found at specified path', 'red'))
        else:
            # build clicks and ads rates dataframe
            self.agg_click_and_ads_rates_dataframe = self._build_agg_mongo_cat_dataframe()

    def agg_click_save_to_csv(self, csv_path=None):
        """ save aggregated data by given path (csv)"""
        if csv_path is None:
            csv_path = self.saving_path_mongo_df
        self.agg_click_and_ads_rates_dataframe.to_csv(self.saving_path_mongo_df,
                                                      index_label=False)
        print(colored(f'agg click and ads rate dataframe is saved to the given path : {self.saving_path_mongo_df}',
                      'green'))


    # building collection from mongo
    def _build_mongo_collection(self):
        client = MongoClient(self.connect_params['connection_string'])
        db = client[self.connect_params['db_name']]
        return db[self.connect_params['collection_name']]
        
    
    def _build_default_mongo_dataframe(self) -> pd.DataFrame:
        def_dataframe = pd.DataFrame(list(self.collection_clicks.find()))
        self.industries_list = list(def_dataframe.Industry.unique())
        # def_dataframe.to_csv('tsms_data/mongo/default_mongo.csv', index_label=False)

        return def_dataframe

    # transform default dataframe by clicks
    def _build_transformed_dataframe(self) -> pd.DataFrame:
        default_df = self._build_default_mongo_dataframe()
        
        dataframes_list = list()
        checking_counter = 0
        pos_checking_counter = 0
        
        print(colored('building mongo data by clicks', 'cyan'))
        for row_number in tqdm(range(default_df.shape[0])):
            current_all_ctns = default_df.Ctns.iloc[row_number]
            current_pos_ctns = default_df.PosCtns.iloc[row_number]
            current_shape = len(current_all_ctns)
        
            current_dataframe = pd.DataFrame(
                {
                    'ctn': pd.Series(default_df.iloc[row_number].Ctns),
                    'Url': [default_df.iloc[row_number].Url]*current_shape,
                    '_id' : [str(default_df.iloc[row_number]._id)]*current_shape,
                    'CampaignId' : [default_df.iloc[row_number].CampaignId]*current_shape,
                    'Industry': [default_df.iloc[row_number].Industry]*current_shape,
                    'clicked': [0]*current_shape 
                }
            )
            current_dataframe['clicked'][current_dataframe.ctn.isin(current_pos_ctns)] = 1
            dataframes_list.append(current_dataframe)
            
            checking_counter += current_shape
            pos_checking_counter += len(current_pos_ctns)
        
        cat_mongo_dataframe = pd.concat(dataframes_list)
        
        del dataframes_list
        del default_df
        del current_dataframe

        # cat_mongo_dataframe.to_csv('tsms_data/mongo/transform_mongo_df.csv',
        #                            index_label=False)
        if self.train_size is not None:
            grouped_filtered_df = cat_mongo_dataframe.groupby(by='ctn').agg('sum').reset_index()[['ctn', 'clicked']]
            non_clicked_size = self.train_size - grouped_filtered_df[grouped_filtered_df['clicked'] > 0].shape[0]

            filtered_ctns = list(grouped_filtered_df[grouped_filtered_df.clicked == 0].sample(non_clicked_size).ctn) + \
                            list(grouped_filtered_df[grouped_filtered_df.clicked > 0].ctn)
            del grouped_filtered_df
            return cat_mongo_dataframe[cat_mongo_dataframe.ctn.isin(filtered_ctns)]
        return cat_mongo_dataframe
    
    # TODO 
    # GROUPING CLICK AND ADS RATE#
    # @staticmethod
    # def _rate_grouping(self, agg='mean'):
    #     pass
            
    def _build_agg_mongo_cat_dataframe(self) -> pd.DataFrame:
        print('start build mongo cat dataframe')
        # mongo_dataframe by clicks
        mongo_cat_df = self._build_transformed_dataframe()
        print(colored('dataframe from Mongo data by clicks | OK', 'green'))
       
        # Industry click rate grouping # time: 12.9 s
        ind_click_rate = pd.DataFrame(
            mongo_cat_df.groupby(
                ['ctn', 'Industry'])['clicked'].agg('mean')).reset_index(drop=False)
        ind_click_rate.rename(columns={'clicked':'click_rate'}, inplace=True)
        print(colored('click rate grouping dataframe | OK', 'green'))
        
        # Industry ads rate grouping # time: 12.2 s
        ind_ad_rate = pd.DataFrame(
            mongo_cat_df.groupby(
                ['ctn', 'Industry'])['clicked'].agg('count')).reset_index(drop=False)
        ind_ad_rate.rename(columns={'clicked':'ad_rate'}, inplace=True)
        print(colored('ads rate grouping dataframe | OK', 'green'))
        
        del mongo_cat_df
        
        # click and ads rates dummies # time: 6.54 s
        ind_dummies_click_rate = pd.get_dummies(ind_click_rate.Industry, prefix='click_rate')
        ind_dummies_ad_rate = pd.get_dummies(ind_ad_rate.Industry, prefix='ad_rate')
        print(colored('click and ads rates dummies | OK', 'green'))
        
        # combine source data with rates dummies # time: 727 ms
        ind_combine = pd.concat([ind_ad_rate, ind_click_rate['click_rate'], ind_dummies_ad_rate, ind_dummies_click_rate], axis=1) 
        print(colored('combine source data with rates dummies | OK', 'green'))

        

        # sum the rows by click_rate # time: 16.0 s
        for col in tqdm(list(ind_dummies_click_rate.columns)):
            ind_combine[col] = ind_combine[col] * ind_combine['click_rate']
        # sum the rows by ad_rate # time: 17.0 s
        for col in tqdm(list(ind_dummies_ad_rate.columns)):
            ind_combine[col] = ind_combine[col] * ind_combine['ad_rate']
        print(colored('sum the rows by click and ads rate | OK', 'green'))
            
        # union rows ads and click rate' to one row # time: 1min 52s
        union_combined_df = ind_combine.groupby('ctn').agg('sum')
        union_combined_df.reset_index(inplace=True)
        print(colored('combine and grouping click and ads rates | OK', 'green'))
        
        del ind_click_rate
        del ind_ad_rate
        del ind_dummies_click_rate
        del ind_dummies_ad_rate
        del ind_combine
        
        return union_combined_df
        
    # slice by current input Industry
    def slice_ind_dataframe(self, industry_name: str):
        agg_dataframe = self.agg_click_and_ads_rates_dataframe[
            ['ctn',
            f'ad_rate_{industry_name}',
            f'click_rate_{industry_name}']]
        agg_ind_ctns_list = list(
            agg_dataframe[
                agg_dataframe[f'ad_rate_{industry_name}'] > 0].ctn.unique())
        return agg_dataframe[agg_dataframe.ctn.isin(agg_ind_ctns_list)], agg_ind_ctns_list
