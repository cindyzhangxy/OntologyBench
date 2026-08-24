from sklearn.model_selection import train_test_split

def make_splits(rows, test_size=0.2,val_size=0.1, seed=42):
    """
    Standardized train/val/test split for all OntologyBench tasks.
    """
    train, test = train_test_split(rows, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=val_size, random_state=seed)
    return train, val, test