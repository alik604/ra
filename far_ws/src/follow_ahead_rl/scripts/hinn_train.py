from time import sleep
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

import catboost
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

import pickle


TRAIN_MLP = False 
EPOCHS = 550 # 400
BATCH_SIZE = 64

TRAIN_CATBOOST = False
TRAIN_RFR = True

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Torch will use {device}')
    
    state_dim = 43
    action_dim = 2

    save_local_1 = './model_weights/HumanIntentNetwork/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentNetwork/list_of_human_state_next.csv'

    list_of_human_state = pd.read_csv(save_local_1).values.tolist()
    list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    if TRAIN_MLP:
        model = HumanIntentNetwork(inner=128, input_dim=state_dim, output_dim=3)
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

            if epoch % 100 == 0:
                model.save_checkpoint()
            if epoch == 400:
                BATCH_SIZE = int(BATCH_SIZE/2)
                optimizer.param_groups[0]['lr'] = 0.0001
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

        CBR = catboost.CatBoostRegressor(iterations = 5000, depth = 8, learning_rate = 0.03, l2_leaf_reg = 0.2, loss_function = "MultiRMSE", thread_count = 6, use_best_model=True, early_stopping_rounds=500) 
        CBR.load_model(PATH)

        # #### fit hardcoded model #####
        CBR.fit(X_train, y_train, eval_set=eval_dataset)
        y_pred = CBR.predict(X_test) # eval set. eval & test are the same

        mse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'MSE is {mse:.6f}')

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
        PATH = './model_weights/HumanIntentNetwork/RandomForestRegressor'
        X_train, X_test, y_train, y_test = train_test_split(list_of_human_state, list_of_human_state_next, test_size=0.00001, random_state=0)
        
        # if os.path.isfile(PATH):
        #   pass

        # #### fit hardcoded model #####
        # regr = pickle.load(open(PATH, 'rb'))
        # regr = RandomForestRegressor(n_estimators=100, max_depth=100, max_features=None, n_jobs=4, random_state=0)
        # regr.fit(X_train, y_train)

        # y_pred = regr.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # print(f'MSE is {mse:.6f}')

        # pickle.dump(regr, open(PATH, 'wb'))

        # #### grid search #####
        regr = RandomForestRegressor()
        random_grid = {'n_estimators': [500], #, 500, 1000, 1500],
                        'max_features': ['auto', None],
                        'max_depth': [None, 100],
                        'min_samples_split': [2, 3],
                        'min_samples_leaf': [3, 4],
                        'bootstrap': [True]}
        # {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
        # {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100}
        rf = RandomForestRegressor()
        rf_random = GridSearchCV(estimator = regr, param_grid = random_grid, cv = 2, verbose=2, n_jobs = -1)
        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)