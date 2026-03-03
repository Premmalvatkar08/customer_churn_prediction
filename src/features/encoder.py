from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    """
    Label encodes categorical features:
    - contract_type
    - internet_service
    """

    categorical_features = ['contract_type', 'internet_service']
    le = LabelEncoder()

    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    return df