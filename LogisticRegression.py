import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from MinMaxClassification.simulation import DataPreparation, ModelTraining, Simulation
from MinMaxClassification.obj_functions_classifiers import logisticRegression_objective_function

import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def split_train_test_and_get_features(df):
    train_df = df[0:round(0.7 * len(df))].copy()
    train_df = DataPreparation(train_df).run_all()

    test_df = df[len(train_df):].copy()
    test_df = DataPreparation(test_df).run_all()

    all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'normalized_price',
                    'rolling_mean_5', 'rolling_mean_10', 'rolling_mean_21', 'angular_coef_5', 'angular_coef_10',
                    'angular_coef_21']
    target = 'target'

    selector = SelectKBest(f_classif, k=9)
    selected = selector.fit_transform(train_df[all_features], train_df[target])
    f_classif_features = selector.get_feature_names_out(all_features)

    selector = SelectKBest(mutual_info_classif, k=9)
    selected = selector.fit_transform(train_df[all_features], train_df[target])
    mutual_info_features = selector.get_feature_names_out(all_features)

    features = [all_features, f_classif_features, mutual_info_features]

    return train_df, test_df, features


stocks = ['PETR4', 'VALE3', 'ITUB4', 'MGLU3', 'PRIO3', 'BBAS3', 'BBDC4', 'ELET3', 'B3SA3', 'RENT3']
lr_time_money_test = []
lr_time_money_train = []
lr_money_earned_train = []
lr_money_earned_test = []
for stock in stocks:
    print(f'Training: {stock}')

# data
    stock = yf.Ticker(f'{stock}.SA')
    df = stock.history(period='max')

    train_df, test_df, features = split_train_test_and_get_features(df)

# model
    clf = LogisticRegression(class_weight='balanced', max_iter=500)

    clf, feature_subset = ModelTraining(train_df, clf, features,
                                        objective_function=logisticRegression_objective_function,
                                        scoring='roc_auc_ovo', cross_val_iterations=1, optimize_trials=2,
                                        scale_data=True).return_model()

# backtest
    money_train, trades_train, values_train = Simulation(train_df[feature_subset], train_df['target'], clf,
                                                         scale_data=True).run()
    money_test, trades_test, values_test = Simulation(test_df[feature_subset], test_df['target'], clf,
                                                      scale_data=True).run()

    lr_money_earned_train.append(money_train)
    lr_money_earned_test.append(money_test)
    lr_time_money_train.append(values_train)
    lr_time_money_test.append(values_test)

    print('------------------------------------------------------------------------------------------------------')

data = {'Stock':stocks, 'Train_set':lr_money_earned_train, 'Test_set':lr_money_earned_test}
final_result = pd.DataFrame(data=data)
print(final_result)