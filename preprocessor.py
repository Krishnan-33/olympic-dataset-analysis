import pandas as pd


def preprocess(df, region_df):
    df = df.merge(region_df, on='NOC', how='left')

    df.drop_duplicates(inplace=True)

    if 'Medal' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)

        df.drop(columns=['notes'], inplace=True)

    return df
