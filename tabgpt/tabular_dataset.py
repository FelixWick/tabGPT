import torch
import numpy as np
from torch.utils.data import Dataset
import os


class TabularDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])

        data = torch.tensor(np.load(file_path))

        # Extract label from the file name or other source
        _, target = self.extract_info_from_filename(self.file_names[idx])

        return data, target

    def extract_info_from_filename(self, filename):
        dataset_name, target, _ = filename.split(";")
        return dataset_name, torch.tensor(float(target), dtype=torch.float32)
    

def load_datasets(dataset_loader, mode):
    datasets = []
    for d in dataset_loader:
        ds_dir = os.path.join(d.current_dir, 'files', mode)
        if os.path.exists(ds_dir):
            datasets.append(TabularDataset(ds_dir))
        else:
            print(f"Warning: Directory for dataset {d.name} does not exist.")
    return datasets


def shuffle(input_embeds):
    # Step 1: Separate the first element (index 0) from the rest (index 1 to 9)
    first_element = input_embeds[:, 0:1, :]  # Shape: (batch_size, 1, 100)
    rest_elements = input_embeds[:, 1:, :]    # Shape: (batch_size, 9, 100)

    # Step 2: Shuffle the rest of the elements along dim 1
    shuffled_indices = torch.randperm(rest_elements.size(1))  # Random permutation of indices for dim=1
    shuffled_rest = rest_elements[:, shuffled_indices, :]      # Shuffle the elements using these indices

    # Step 3: Concatenate the first element with the shuffled rest
    return torch.cat((first_element, shuffled_rest), dim=1)