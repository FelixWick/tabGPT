
from tabgpt.col_embed import Embedder
from tabgpt.data.house_prices.data_setup import HousePricesData
from tabgpt.data.store_sales.data_setup import StoreSalesData
from tabgpt.model_hf import tabGPT_HF, tabGPTConfig
from tabgpt.utils import predict
import torch
from torch.utils.data import TensorDataset, DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


if __name__ == '__main__':

    # load in data
    store_sales = StoreSalesData()
    store_sales.setup()
    n_cols = store_sales.n_features

    target_column = store_sales.main_target
    target_scaler = store_sales.scaler[target_column]

    max_length = store_sales.n_features + 1


    n_layer, n_head = 4, 4 # gpt-micro
    config = tabGPTConfig(n_layer=n_layer, n_head=n_head, block_size=max_length, n_output_nodes=1)
    model = tabGPT_HF(config)

    # load in pretrained model
    model = model.from_pretrained('store_sales_model').to(device)

    embedder = Embedder(store_sales)
    
    # load in test-set
    store_sales.test_setup()

    # construct embeddings
    embedder.test()
    features_embeds_test = embedder.embed(n_cols=store_sales.n_features)

    # build test-dataset
    test_dataset = TensorDataset(
        features_embeds_test,
        torch.tensor(store_sales.df_test[store_sales.main_target].tolist(), dtype=torch.float32)
    )

    df_test = store_sales.df_test
    
    df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test, target_scaler)
    
    print('Saving submission file')
    df_test[["id", "yhat"]].rename(columns={"yhat": target_column}).to_csv("submission.csv", index=False)