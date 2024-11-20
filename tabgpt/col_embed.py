from tabgpt.data_loader import DataFrameLoader
import torch
import numpy as np
from transformers import GPT2Model, AutoTokenizer
import numpy as np
import logging
import os
import glob
import shutil
import json

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


class Embedder():

    def __init__(self, data_frame_loader: DataFrameLoader, mode=None, device='cuda'):
        self.df_loader = data_frame_loader
        self.model = GPT2Model.from_pretrained('gpt2').to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if mode is not None:
            self.df_loader.mode = mode
        
    def train(self):
        self.df_loader.mode = 'train'

    def val(self):
        self.df_loader.mode = 'val'

    def test(self):
        self.df_loader.mode = 'test'


    def embed(self, n_cols, null_treatment="zero-embedding", fillna_categorical="missing value", fillna_numerical=0, save=False, remove_first=True):

        if save == False:
            if len(self.df_loader.target_column) > 1:
                raise ValueError('Specifying multiple target columns for in-memory mode is not allowed. Either use save=True or choose only one main target.')
        n_features = self.df_loader.n_features + 1 # plus target description
        n_cols += 1 # plus target description
        assert n_features <= n_cols, "total number of features must not be larger than set n_cols"

        df = self.df_loader.df() # train or val depending on mode
        inputs_embeds = torch.empty(len(df), 1, 768)

        categorical_features = self.df_loader.categorical_features
        numerical_features = self.df_loader.numerical_features

        df[categorical_features] = df[categorical_features].fillna(fillna_categorical)
        df[numerical_features] = df[numerical_features].fillna(fillna_numerical)

        column_identifier = {}
        features = categorical_features + numerical_features
        for i, col in enumerate(features):
            column_identifier[col] = i + 1 # due to target description embedding
            input_ids = self.tokenizer(col, return_tensors="pt")
            with torch.no_grad():
                colname_embed = self.model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()
            colname_embed = colname_embed.repeat(len(df), 1)

            if col in categorical_features:
                cat_embed_dict = {}
                for category in df[col].unique().tolist():
                    input_ids = self.tokenizer(str(category), return_tensors="pt")
                    with torch.no_grad():
                        cat_embed_dict[category] = self.model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()

                cat_embeds = torch.stack([cat_embed_dict[val] for val in df[col]])

                col_embeds = colname_embed + cat_embeds.squeeze(1)
            else:
                col_values = torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1)

                if null_treatment == "shift":
                    col_embeds = colname_embed * torch.where(col_values >= 0, col_values + 1, col_values - 1)
                elif null_treatment == "zero-embedding":
                    col_values = torch.where(col_values == 0, 1, col_values)
                    col_embeds = colname_embed * col_values

                    input_ids = self.tokenizer(str(0), return_tensors="pt")
                    with torch.no_grad():
                        cat0_embed = self.model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()

                    mask = (df[col] == 0).values
                    cat_embeds = torch.zeros(len(df), 768)
                    cat_embeds[mask, :] = cat0_embed

                    col_embeds = col_embeds + cat_embeds
                elif "simple":
                    col_embeds = colname_embed * col_values
                else:
                    raise ValueError

            inputs_embeds = torch.cat((inputs_embeds, col_embeds.unsqueeze(1)), dim=1)
        
        inputs_embeds = inputs_embeds[:, 1:, :]

        target_embedding_dict = {}
        targets = {}
        for target_col in self.df_loader.target_column:
            descr = f'{self.df_loader.task_description} - {target_col} prediction'
            logging.info(f"{self.df_loader.name}: Building {self.df_loader.mode} data with description: '{descr}'")
            input_ids = self.tokenizer(descr, return_tensors="pt")
            with torch.no_grad():
                target_embed = self.model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()

            target_embedding_dict[target_col] = column_identifier[target_col], target_embed.tolist()            
            targets[target_col] = column_identifier[target_col], df[target_col].to_list()


        if n_features < n_cols:
            padding_features_embeds = torch.ones(len(df), n_cols - n_features, 768)
            inputs_embeds = torch.cat((inputs_embeds, padding_features_embeds), dim=1)

        if save:
            path = os.path.join(self.df_loader.current_dir,'files',self.df_loader.mode)

            task_description_path = os.path.join(self.df_loader.current_dir,'task_description',self.df_loader.mode)
            if not os.path.exists(task_description_path):
                logging.info("Creating task description directory")
                os.makedirs(task_description_path)
            with open(os.path.join(task_description_path,'task_description.json'), 'w') as f: 
                json.dump(target_embedding_dict, f)

            if not os.path.exists(path):
                logging.info("Creating file directory")
                os.makedirs(path)

            if remove_first:
                if os.listdir(path):
                    files = glob.glob(os.path.join(path, '*'))
                    logging.info('Removing old files')
                    for f in files:
                        try:
                            os.remove(f)
                        except:
                            shutil.rmtree(f)
            
            logging.info(f'Storing {self.df_loader.mode} files')
            for i in range(inputs_embeds.shape[0]):
                target_dict = [(c,t[i]) for _, (c,t) in targets.items()]
                file_name = f"{self.df_loader.name};{target_dict};{i}"
                np.save(os.path.join(path,file_name), inputs_embeds[i].to(torch.float16))
        else:
            n, _, _ = inputs_embeds.shape
            c, target_embed = target_embedding_dict[self.df_loader.main_target]
            target_embed = target_embed.unsqueeze(0).repeat(n,1,1)
            feature_embeds = torch.cat((target_embed,inputs_embeds), dim=1)
            pos_info = torch.zeros_like(feature_embeds)
            pos_info[:, 1:, :] = 1.
            feature_embeds = feature_embeds + pos_info
            feature_embeds = remove_index(feature_embeds, idx=c)
            return feature_embeds


class DirectoryNotEmptyError(Exception):
    def __init__(self, path):
        super().__init__(f"The directory '{path}' is not empty.")


def remove_index(t, idx):
    return torch.cat((t[:, :idx, :], t[:, (idx+1):, :]), dim=1)


def add_positional_info(target_embedding, feature_embeddings):
    """
    For a single row, add the target embedding and positional info
    """
    feature_embeddings_descr = torch.cat((target_embedding, feature_embeddings), dim=0)
    pos_info = torch.zeros_like(feature_embeddings_descr)
    pos_info[1:, :] = 1.
    return feature_embeddings_descr + pos_info