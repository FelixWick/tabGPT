import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import os
import logging


class TabularDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        # Sort the file names by the extracted index
        self.sorted_files = sorted(self.file_names, key=self.extract_index)


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.sorted_files[idx])

        data = torch.tensor(np.load(file_path))

        # Extract label from the file name or other source
        _, target = self.extract_info_from_filename(self.sorted_files[idx])

        return data, target

    def extract_info_from_filename(self, filename):
        dataset_name, target, _ = filename.split(";")
        return dataset_name, torch.tensor(float(target), dtype=torch.float32)

    def extract_index(self,file_name):
        return int(file_name.split(';')[-1].split('.')[0])
    
def join_paths(*paths):
    # Filter out None values
    filtered_paths = [p for p in paths if p is not None]
    return os.path.join(*filtered_paths)

def load_datasets(dataset_loader, mode, target=None, only_main=False):
    datasets = []
    for d in dataset_loader:
        ds_dir = os.path.join(join_paths(d.current_dir, 'files', mode, target))
        if target is None and not only_main:
            for i, target_col in enumerate(os.listdir(ds_dir)):
                logging.info(f'Creating {mode} dataset {i} for {target_col}')
                datasets.append(TabularDataset(os.path.join(ds_dir,target_col)))
        elif target:
            logging.info(f'Creating {mode} dataset for user-defined {target}')
            datasets.append(TabularDataset(ds_dir))
        else:
            logging.info(f'Creating {mode} dataset for main-target {d.main_target}')
            datasets.append(TabularDataset(os.path.join(ds_dir,d.main_target)))
            
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