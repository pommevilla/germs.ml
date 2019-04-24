def train_val_test_split(attributes, targets, seed = 489):
    """
        Splits a a dataset into training, validation, and testing sets.
        Automatically does a 60/20/20 split.
    """
    from sklearn.model_selection import train_test_split
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        attributes, targets, test_size = 0.2, random_state = seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size = 0.25, random_state = seed
    )
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_missing_percentage(df):
    import pandas as pd
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    return missing_value_df
