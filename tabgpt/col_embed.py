from tabgpt.data_loader import DataFrameLoader
import torch
import numpy as np
from transformers import GPT2Model, AutoTokenizer
import numpy as np
import logging
import os
import glob

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
        n_features = self.df_loader.n_features + 1
        n_cols += 1
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

        for target_col in self.df_loader.target_column:
            descr = f'{self.df_loader.task_description} - {target_col} prediction'
            logging.info(f"{self.df_loader.name}: Building {self.df_loader.mode} data with description: '{descr}'")
            input_ids = self.tokenizer(descr, return_tensors="pt")
            with torch.no_grad():
                target_embed = self.model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()
            target_embed = target_embed.repeat(len(df), 1, 1) # nrows, 1, emb_dim

            features_embeds_wo_pos = torch.cat((target_embed, inputs_embeds), dim=1)
            rows, features, emb_dim = features_embeds_wo_pos.shape
            features_embeds = torch.zeros(rows, features, emb_dim)
            features_embeds[:, 1:, :] = 1.
            features_embeds += features_embeds_wo_pos

            features_embeds = remove_index(features_embeds, column_identifier[target_col])
            target = df[target_col].to_list()
            # loss_identifier = 0 if type=='binary' else 1

            if n_features < n_cols:
                padding_features_embeds = torch.ones(len(df), n_cols - n_features, 768)
                features_embeds = torch.cat((features_embeds, padding_features_embeds), dim=1)

            if save:
                path = os.path.join(self.df_loader.current_dir,'files',self.df_loader.mode, target_col)

                if not os.path.exists(path):
                    logging.info("Creating directory")
                    os.makedirs(path)

                if remove_first:
                    if os.listdir(path):
                        files = glob.glob(os.path.join(path, '*'))
                        logging.info('Removing old files')
                        for f in files:
                            os.remove(f)
                
                logging.info(f'Storing {self.df_loader.mode} files with target: {target_col}')
                for i in range(features_embeds.shape[0]):
                    file_name = f"{self.df_loader.name};{target[i]};{i}"
                    np.save(os.path.join(path,file_name), features_embeds[i].to(torch.float16))
            else:
                return features_embeds


class DirectoryNotEmptyError(Exception):
    def __init__(self, path):
        super().__init__(f"The directory '{path}' is not empty.")


def remove_index(t, idx):
    return torch.cat((t[:, :idx, :], t[:, (idx+1):, :]), dim=1)
