from operator import itemgetter
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LassoCV
import xgboost as xgb

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from .sampling import select_random_params
from ..models.benchmark import OnetoNet, MMDLoss
from ..models.neural_fitters import TabularDataSet, BaseMLP

def train_boosting_model(data, hyperparam_selction, num_trials=15):

    X_train_, X_val_, y_train_, y_val = train_test_split(data['X_train'],
                                                        data['y_train'], 
                                                        test_size=0.2,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train_, label=y_train_)
    dval = xgb.DMatrix(X_val_, label=y_val)

    dtrain_complete = xgb.DMatrix(data['X_train'], label=data['y_train'])
    dtest = xgb.DMatrix(data['X_test'])

    hyperparam_opt_s = []
    for trial_ in range(num_trials):
        param_choice = select_random_params(hyperparam_selction)
        param_choice['objective'] = 'reg:squarederror'

        model = xgb.train(
                        param_choice,
                        dtrain,
                        num_boost_round=250,
                        evals=[(dval, "Validation")],
                        early_stopping_rounds=10, 
                        verbose_eval=False)
        
        intermed_results = {'score': model.best_score, 
                            'iterations': model.best_iteration, 
                            'params': param_choice
                            }

        hyperparam_opt_s.append(intermed_results)

    scores_all = [key_['score'] for key_ in hyperparam_opt_s]
    index, _ = min(enumerate(scores_all), key=itemgetter(1))

    params_best = hyperparam_opt_s[index]['params']
    iterations_best = hyperparam_opt_s[index]['iterations']

    # Train best model on all data
    model = xgb.train(params_best,
                      dtrain_complete,
                      num_boost_round=iterations_best)

    predictions = model.predict(dtest)

    return predictions

def train_linear_models(data, cv_iters=10):
    model = (MultiOutputRegressor(LassoCV(cv=cv_iters, random_state=42))
            .fit(data['X_train'], data['y_train']))
    
    predictions = model.predict(data['X_test'])

    return predictions

def train_fix_onetonets(data, hyper_params, sens_feature_idx, data_dict, tries_=5):

    data_train_all = TabularDataSet(data['X_train'],
                            np.float32(data['y_train'][:,0]).reshape(-1,1),
                            np.float32(data['y_train'][:,1]).reshape(-1,1))
    trainloader_all = DataLoader(data_train_all, batch_size=128, shuffle=True)
    
    
    all_res = []
    # set up architecture
    d = data_dict['X_train'].shape[1]
    lambda_ = 10e-5
    gamma_=2
    r_=2^3*d

    model = OnetoNet(architecture=r_,
                    input_dim=data_dict['X_train'].shape[1],
                    n_tasks= 2,
                    activation='relu')

    optimizer = optim.Adam(model.parameters(),
                        lr=0.05)
        
    for epoch in range(75):
        for i, (x_train, y1, y2) in enumerate(trainloader_all):
            optimizer.zero_grad()
            W_, V_ = model(x_train)

            sens_w_1 = W_[x_train[:,sens_feature_idx] == 1, :]
            sens_w_0 = W_[x_train[:,sens_feature_idx] == 0, :]

            loss_fct_1 = nn.MSELoss()
            loss_fct_2 = nn.MSELoss()
            mmd_loss = MMDLoss()

            loss_1 = loss_fct_1(V_[:,0], y1.squeeze())
            loss_2 = loss_fct_2(V_[:,1], y2.squeeze())
            mmd_ = mmd_loss(sens_w_0, sens_w_1)

            F_V = torch.norm(model.layers['output'].weight[0], p='fro')
            F_W = torch.norm(model.layers['embedding'].weight[0], p='fro')
            
            loss = loss_1 + loss_2 + lambda_*(F_V + F_W)  + gamma_*mmd_ 

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        _, preds = model(torch.from_numpy(data['X_test']))

    return preds.detach().numpy()
