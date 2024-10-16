from tabgpt.col_embed import add_positional_info, remove_index
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import os
import logging
from IPython import embed
import json



class TabularDataset(Dataset):
    def __init__(self, folder_path, task_description_path,n_targets):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        # Sort the file names by the extracted index
        self.sorted_files = sorted(self.file_names, key=self.extract_index)

        self.n_targets = n_targets

        with open(os.path.join(task_description_path,'task_description.json'), 'r') as f:
            self.task_dict = json.load(f)

        self.cache_files = {}

    def __len__(self):
        return len(self.file_names) * self.n_targets

    
    def _load_file(self, file_path):
        if self.cache_files:
            # Load and cache the file if it's not already cached
            if file_path not in self.file_cache:
                data = torch.tensor(np.load(file_path))
                self.file_cache[file_path] = data
            return self.file_cache[file_path]
        else:
            # Load the image without caching
            return torch.tensor(np.load(file_path))

    def __getitem__(self, idx):
        actual_idx = idx // self.n_targets
        augment_idx = idx % self.n_targets
        file_path = os.path.join(self.folder_path, self.sorted_files[actual_idx])

        data = self._load_file(file_path)

        # Extract label from the file name or other source
        _, target_list = self.extract_info_from_filename(self.sorted_files[actual_idx])

        target = torch.tensor(target_list[augment_idx][1],dtype=data.dtype)
        data = self.prepare_data(data, augment_idx)

        return data, target
    
    def prepare_data(self, data, augment_idx):
        keys = list(self.task_dict.keys())
        c, target_embedding = self.task_dict[keys[augment_idx]]
        target_embedding = torch.tensor(target_embedding,dtype=data.dtype)
        data = add_positional_info(target_embedding=target_embedding, feature_embeddings=data)
        return torch.cat((data[:c, :], data[(c+1):, :]), dim=0)

    def extract_info_from_filename(self, filename):
        dataset_name, target_list, _ = filename.split(";")
        target_list = eval(target_list)
        return dataset_name, target_list

    def extract_index(self,file_name):
        return int(file_name.split(';')[-1].split('.')[0])
    
def join_paths(*paths):
    # Filter out None values
    filtered_paths = [p for p in paths if p is not None]
    return os.path.join(*filtered_paths)

def load_datasets(dataset_loader, mode, only_main=False):
    datasets = []
    for d in dataset_loader:
        ds_dir = os.path.join(join_paths(d.current_dir, 'files', mode))
        datasets.append(TabularDataset(folder_path=ds_dir, 
                                       task_description_path=os.path.join(d.current_dir, 'task_description', mode),
                                       n_targets=len(d.target_column) if not only_main else 1))
            
    return ConcatDataset(datasets)


def shuffle(input_embeds):
    # Step 1: Separate the first element (index 0) from the rest (index 1 to 9)
    first_element = input_embeds[:, 0:1, :]  # Shape: (batch_size, 1, 100)
    rest_elements = input_embeds[:, 1:, :]    # Shape: (batch_size, 9, 100)

    # Step 2: Shuffle the rest of the elements along dim 1
    shuffled_indices = torch.randperm(rest_elements.size(1))  # Random permutation of indices for dim=1
    shuffled_rest = rest_elements[:, shuffled_indices, :]      # Shuffle the elements using these indices

    # Step 3: Concatenate the first element with the shuffled rest
    return torch.cat((first_element, shuffled_rest), dim=1)