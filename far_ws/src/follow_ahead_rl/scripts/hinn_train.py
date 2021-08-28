from time import sleep
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# import gym
# import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

import catboost
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
import pickle, os

EPOCHS = 85 # 400
BATCH_SIZE = 128
TRAIN_MLP = False 
TRAIN_CATBOOST = True
TRAIN_RFR = True
TRAIN_POLY = True

'''
Notes: Dataset was made with `USE_TESTING = False` in gym_gazeboros_ac.py. That means the robot spawns at constant locations

'''

if __name__ == '__main__':

    # save_local_1 = './model_weights/HumanIntentNetwork/Saves/list_of_human_state.csv'
    # save_local_2 = './model_weights/HumanIntentNetwork/Saves/list_of_human_state_next.csv'
    # list_of_human_state = pd.read_csv(save_local_1).values.tolist()
    # list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    save_local= './model_weights/HumanIntentNetwork/Saves/Human_xyTheta_ordered_triplets_0.pickle'
    save_local_no_dup= './model_weights/HumanIntentNetwork/Saves/Human_xyTheta_ordered_triplets_no_duplicates.pickle'
    DROP_DUPLICATES=True
    # tmp = pd.read_csv(save_local).values#.tolist()
    if os.path.isfile(save_local):
        with open(save_local, 'rb') as handle:
            tmp = pickle.load(handle)
            tmp = np.asarray(tmp)
            # for i in range(len(tmp)): , dtype=np.float32
            #     tmp[i] = np.array(tmp[i])
            if DROP_DUPLICATES:
                print(f'before drop duplicates shape  {tmp.shape}')
                tmp = np.unique(tmp, axis=0)
                print(f'after drop duplicates shape {tmp.shape}')
                with open(save_local_no_dup, 'wb') as handle:
                    pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise Exception(f"Warning: Tried to load previous data but files were not found!\nLooked in location {save_local}")




    tmp = tmp[:20000000] # 20000000 runs, not testing on MLP

    print(f'tmp[0:5]\n{tmp[0:5]}')
    print(f'tmp.shape {tmp.shape}')

    print(f'tmp[1][0,:] {tmp[1][0,:]}')
    print(f'target col.shape {tmp[:,0,:].shape}')
    print(f'data   col.shape {tmp[:,1:,:].shape}')

    list_of_human_state = tmp[:,1:,:]#.astype(np.float32)
    list_of_human_state_next = tmp[:,0,:]#.astype(np.float32)

    print(f'list_of_human_state[:5]\n{list_of_human_state[:5]}')
    print(f'list_of_human_state_next[:5]\n{list_of_human_state_next[:5]}')

    flatten_dim = list_of_human_state.shape[1] * list_of_human_state.shape[2]
    list_of_human_state = list_of_human_state.reshape(-1, flatten_dim)
    print(f'[after flatten] list_of_human_state[:5]\n{list_of_human_state[:5]}')

    # exit()
    if TRAIN_MLP:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        print(f'Torch will use {device}')

        state_dim = len(list_of_human_state[0])# 43
        output_dim = len(list_of_human_state_next[0]) # 3
        print(f'State_dim {state_dim} | output_dim {output_dim}')
        model = HumanIntentNetwork(inner=64, input_dim=state_dim, output_dim=output_dim)
        model.load_checkpoint()
        model.to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam((model.parameters()), lr=1e-3)

        human_state_tensor = torch.Tensor(list_of_human_state).to(device)
        next_human_state_tensor = torch.Tensor(list_of_human_state_next).to(device)

        losses = []
        for epoch in range(EPOCHS):
            _sum = 0
            for i in range(0, human_state_tensor.size(0), BATCH_SIZE):
                target = next_human_state_tensor[i: i+BATCH_SIZE]
                input_batch = human_state_tensor[i: i+BATCH_SIZE]
                pred = model.forward(input_batch)

                optimizer.zero_grad()
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                _sum += loss.item()

                # print(f'pred {pred}')
                # print(f'target {target}')

            if epoch % 10 == 0:
                model.save_checkpoint()
            if epoch == 25: # 400
                BATCH_SIZE = max(int(BATCH_SIZE/2), 1)
                optimizer.param_groups[0]['lr'] *= 0.5 # = 0.0001
                print(f'\tBatch size is now {BATCH_SIZE} and the LR is now {0.0001}')
                
            losses.append(_sum)
            print(f'Epoch {epoch} | Loss_sum {_sum:.4f}')

        model.save_checkpoint()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss of (possibly) Pre-Trained model')
        plt.savefig('image.png')
        plt.show()

        print("END")

    if TRAIN_CATBOOST: 
        PATH = './model_weights/HumanIntentNetwork/CatBoostRegressor'
        X_train, X_test, y_train, y_test = train_test_split(list_of_human_state, list_of_human_state_next, test_size=0.1, random_state=0)
        train_dataset = catboost.Pool(X_train, y_train) 
        eval_dataset = catboost.Pool(X_test, y_test)

        CBR = catboost.CatBoostRegressor(iterations = 1500, depth = 8, learning_rate = 0.03, l2_leaf_reg = 0.2, loss_function = "MultiRMSE", thread_count = 6, use_best_model=True) # , early_stopping_rounds=500
        if os.path.isfile(PATH):
            CBR.load_model(PATH)
            print(f'loading Catboost model...')

        # #### fit hardcoded model #####
        CBR.fit(X_train, y_train, eval_set=eval_dataset)
        y_pred = CBR.predict(X_test) # eval set. eval & test are the same

        mse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'MSE for CatBoostRegressor is {mse:.6f}\n')

        CBR.save_model(PATH)

        # #### grid search #####
        # grid = {'iterations': [800],
        #         'learning_rate': [0.1, 0.2, 0.3, 0.4],
        #         'depth': [8],
        #         'l2_leaf_reg': [0.1, 0.2, 0.3, 0.5]
        #         }
        # train_dataset = catboost.Pool(list_of_human_state, list_of_human_state_next) 
        # grid_search_results = CBR.grid_search(grid, train_dataset, cv=2, train_size=0.8) 
        # # {'depth': 8, 'iterations': 1000, 'learning_rate': 0.03, 'l2_leaf_reg': 0.2}
        # # {'depth': 8, 'iterations': 1000, 'learning_rate': 0.1, 'l2_leaf_reg': 0.3} # l2_leaf_reg 0.1 or 0.2 is best. not 0.3
        # print(grid_search_results)
        # print(grid_search_results['params'])

    if TRAIN_RFR:
        from sklearn.ensemble import RandomForestRegressor

        PATH = './model_weights/HumanIntentNetwork/RandomForestRegressor'
        X_train, X_test, y_train, y_test = train_test_split(list_of_human_state, list_of_human_state_next, test_size=0.1, random_state=0)
        
        if os.path.isfile(PATH):
            print(f'RandomForestRegressor save found. Skipping training...')
            regr = pickle.load(open(PATH, 'rb'))
            # print(regr.best_params_) # only for girdsearch obj
        else:
            print(f'RandomForestRegressor save not found. Training...')

            #### fit hardcoded model #####
            
            regr = RandomForestRegressor(n_estimators=100, max_depth=100, max_features=None, n_jobs=4, random_state=0)
            regr.fit(X_train, y_train)
            pickle.dump(regr, open(PATH, 'wb'))
        y_pred = regr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE for RandomForestRegressor is {mse:.12f}\n')

            

        # #### grid search #####
        # regr = RandomForestRegressor()
        # random_grid = {'n_estimators': [500], #, 500, 1000, 1500],
        #                 'max_features': ['auto', None],
        #                 'max_depth': [None, 100],
        #                 'min_samples_split': [2, 3],
        #                 'min_samples_leaf': [3, 4],
        #                 'bootstrap': [True]}
        # {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
        # {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100}
        # rf_random = GridSearchCV(estimator = regr, param_grid = random_grid, cv = 2, verbose=2, n_jobs = -1)
        # rf_random.fit(X_train, y_train)
        # print(rf_random.best_params_)

    if TRAIN_POLY:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        PATH = './model_weights/HumanIntentNetwork/PolynomialRegressor'
        # X_train, X_test, y_train, y_test = train_test_split(list_of_human_state, list_of_human_state_next, test_size=0.1, random_state=0)
        idx = int(len(list_of_human_state)*0.9)
        X_train, X_test, y_train, y_test = list_of_human_state[:idx], list_of_human_state[idx:], list_of_human_state_next[:idx], list_of_human_state_next[idx:]

        poly_reg = PolynomialFeatures(degree=2)
        X_poly_train = poly_reg.fit_transform(X_train)
        print(f'X_train.shape {np.array(X_train).shape}')
        print(f'X_poly_train.shape {X_poly_train.shape}')
        
        if os.path.isfile(PATH):
            print(f'PolynomialRegressor save found. Skipping training...')
            regr = pickle.load(open(PATH, 'rb'))
            # print(f'coef: {regr.coef_} | intercept: {regr.coef_}')
        else:
            print(f'PolynomialRegressor save not found. Training...')
            regr = LinearRegression()
            regr.fit(X_poly_train, y_train)
            pickle.dump(regr, open(PATH, 'wb'))
            #### fit hardcoded model #####
            # degree=1  | MSE is 0.124031
            # degree=2  | MSE is 0.118514
            # degree=3  | MSE is 0.134984  |  5000 random features

            # X_poly_train = np.array(X_poly_train)
            # size = X_poly_train.shape[1]
            # idx = np.random.random_integers(0, size, 5000)
            # X_poly_train = X_poly_train[:, idx]
            # print(f'X_poly_train {X_poly_train.shape}')

        y_pred = regr.predict(poly_reg.transform(X_test)) # [:, idx]
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE for PolynomialRegressor is {mse:.12f}\n')
    
        def visualization(start, end, X_test=X_test, y_test=y_test):
            y_pred = regr.predict(poly_reg.transform(X_test[start:end])) 
            y_test = np.array(y_test)
            # print(y_test[start:end])
            print(y_test[start:end, 0])
            plt.figure(figsize=(10, 10))
            plt.plot(y_pred[start:end,0], y_pred[start:end,1], color='r', label='x,y of y_pred')
            plt.plot(y_test[start:end,0], y_test[start:end,1], color='k', label='x,y of X_test')
            plt.show()

        # print(f'y_test\n{np.array(y_test)[-100:]}')
        #### Visualization ####
        # visualization(0, 120)
        # visualization(500, 520)

