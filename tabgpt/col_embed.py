import numpy as np
import torch

from sentence_transformers import SentenceTransformer


def get_column_embeddings(df, target_name, categorical_features, numerical_features,
                          number_of_cols=10,
                          null_treatment="zero-embedding", fillna_categorical="missing value", fillna_numerical=0):
    features = categorical_features + numerical_features
    number_of_features = len(features) + 1
    number_of_cols += 1
    assert number_of_features <= number_of_cols, "total number of features must not be larger than set number_of_cols"

    model = SentenceTransformer("all-MiniLM-L6-v2")

    inputs_embeds = np.empty((len(df), 1, 384))

    df[categorical_features] = df[categorical_features].fillna(fillna_categorical)
    df[numerical_features] = df[numerical_features].fillna(fillna_numerical)

    for col in features:
        colname_embed = model.encode(col)
        colname_embed = np.reshape(colname_embed, (1, -1))
        colname_embed = colname_embed.repeat(len(df), 0)

        if col in categorical_features:
            cat_embed_dict = {}
            for category in df[col].unique().tolist():
                cat_embed_dict[category] = model.encode(str(category))

            cat_embeds = []
            for val in df[col]:
                cat_embeds.append(cat_embed_dict[val])
            cat_embeds = np.stack(cat_embeds)

            col_embeds = colname_embed + cat_embeds
        else:
            col_values = np.reshape(df[col].values, (-1, 1))
            if null_treatment == "shift":
                col_embeds = colname_embed * np.where(col_values >= 0, col_values + 1, col_values - 1)
            elif null_treatment == "zero-embedding":
                col_embeds = colname_embed * np.where(col_values == 0, 1, col_values)

                cat0_embed = model.encode(str(0))

                mask = (df[col] == 0).values
                cat_embeds = np.zeros((len(df), 384))
                cat_embeds[mask, :] = cat0_embed

                col_embeds = col_embeds + cat_embeds
            elif "simple":
                col_embeds = colname_embed * col_values
            else:
                raise ValueError

        inputs_embeds = np.concatenate((inputs_embeds, np.reshape(col_embeds, (len(df), 1, -1))), axis=1)
    
    inputs_embeds = inputs_embeds[:, 1:, :]

    target_embed = model.encode(target_name)
    target_embed = np.reshape(target_embed, (1, 1, -1)) # nrows, 1, emb_dim
    target_embed = target_embed.repeat(len(df), 0)

    features_embeds = np.concatenate((target_embed, inputs_embeds), axis=1)

    if number_of_features < number_of_cols:
        padding_features_embeds = np.ones((len(df), number_of_cols - number_of_features, 384))
        features_embeds = np.concatenate((features_embeds, padding_features_embeds), axis=1)

    return torch.tensor(features_embeds, dtype=torch.float32)
