import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from operator import itemgetter

def sample_data_sim(feature_data, label_data, seed, other_cols, numeric_cols, scale_y=False):
    # Encoders and pipeline
    onehot_ = OneHotEncoder(handle_unknown='ignore')
    scaler_ = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[('categoricals', onehot_, other_cols), 
                    ('numerical', scaler_, numeric_cols)
                    ], 
                    remainder='passthrough', 
                    sparse_threshold=0)


    # Split
    X_train, X_test, y_train, y_test = train_test_split(feature_data,
                                                        label_data, 
                                                        test_size=0.2,
                                                        random_state=seed)

    # Transform and save sentitive feature IDX
    transformer.fit(X_train)
    X_train_scaled = np.float32(transformer.transform(X_train))
    X_test_scaled = np.float32(transformer.transform(X_test))

    scalers_dict = {'transformer': transformer}

    check_names = ['SEX' in v for v in list(transformer.get_feature_names_out())]
    idx_sensitive_feature, _ = max(enumerate(check_names), key=itemgetter(1))

    if scale_y:
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)

        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)

        scalers_dict['scaler_y'] =  scaler_y

    ret_dict = {'X_train': X_train_scaled, 
                'y_train': y_train, 
                'X_test': X_test_scaled, 
                'y_test': y_test}
    
    return ret_dict, scalers_dict, idx_sensitive_feature


def select_random_params(param_dict):
    params = {}
    for key, values in param_dict.items():
        hyper_choice = random.choice(list(values))
        params[key] = hyper_choice
    return params