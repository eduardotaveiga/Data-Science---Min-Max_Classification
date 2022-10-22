import numpy as np
from scipy.signal import argrelextrema

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler

import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)



class DataPreparation():

    def __init__(self, df):
        self.df = df

    def normalized_price(self):
        print('Normalized price')

        def np(row):
            cl = row['Close'] - row['Low']
            hl = row['High'] - row['Low']

            if hl == 0:
                return 1.0
            else:
                return cl / hl


        self.df['normalized_price'] = self.df.apply(np, axis=1)

        return self


    def local_max_and_min(self):
        print('Local Max and Min')
        x = np.array(self.df['Close'])

        # for local maxima
        lmax = argrelextrema(x, np.greater, order=15, mode='wrap')

        # for local minima
        lmin = argrelextrema(x, np.less, order=15, mode='wrap')

        labels = np.zeros(len(self.df['Close']))

        idx = 0
        for c in range(0, len(self.df['Close'])):
            if idx in lmin[0]:
                labels[idx] = 0
            elif idx in lmax[0]:
                labels[idx] = 1
            else:
                labels[idx] = 2
            idx += 1

        self.df['target'] = labels
        return self


    def find_new_min(self):
        max_founds = 0
        old = 0
        idx_old = 0
        write_index = True
        idx = 0

        new_points = []
        for c in self.df['target']:
            if c == 1:
                max_founds += 1
                if write_index == True:
                    idx_old = idx
                    write_index = False
            if c == 0:
                max_founds = 0
                write_index = True
            if max_founds == 2:
                # print(idx_old, idx, 'duplicated')

                x = np.array(self.df['Close'][idx_old:idx + 1])
                minp = argrelextrema(x, np.less_equal, order=idx - idx_old)
                try:
                    min_idx = minp[0][0]
                except:
                    continue
                else:
                    new_points.append(idx_old + minp[0][0])

                max_founds = 0
                write_index = True
            idx += 1

        labels = np.zeros(len(self.df['Close']))

        idx = 0
        for c in range(0, len(self.df['Close'])):

            if idx in new_points:
                labels[idx] = 0
            else:
                labels[idx] = self.df['target'].iloc[idx]
            idx += 1

        self.df['target'] = labels
        return self


    def find_new_max(self):
        min_founds = 0
        old = 0
        idx_old = 0
        write_index = True
        idx = 0

        new_points = []
        for c in self.df['target']:
            if c == 0:
                min_founds += 1
                if write_index == True:
                    idx_old = idx
                    write_index = False
            if c == 1:
                min_founds = 0
                write_index = True
            if min_founds == 2:
                # print(idx_old, idx, 'duplicated')

                x = np.array(self.df['Close'][idx_old:idx + 1])
                minp = argrelextrema(x, np.greater_equal, order=idx - idx_old)
                try:
                    min_idx = minp[0][0]
                except:
                    continue
                else:
                    new_points.append(idx_old + minp[0][0])

                min_founds = 0
                write_index = True
            idx += 1

        labels = np.zeros(len(self.df['Close']))

        idx = 0
        for c in range(0, len(self.df['Close'])):

            if idx in new_points:
                labels[idx] = 1
            else:
                labels[idx] = self.df['target'].iloc[idx]
            idx += 1

        self.df['target'] = labels
        return self


    def rolling_mean(self):
        print('Rolling mean')

        def get_angular_coef(df, period):
            coef = []
            for c in range(0, len(df)):
                if c + period >= len(df):
                    a, b = np.polyfit(x=[0, 1], y=[df[f'rolling_mean_{period}'].iloc[c],
                                                   df[f'rolling_mean_{period}'].iloc[len(df) - 1]], deg=1)
                    coef.append(a)
                else:
                    a, b = np.polyfit(x=[0, 1], y=[df[f'rolling_mean_{period}'].iloc[c],
                                                   df[f'rolling_mean_{period}'].iloc[c + period]], deg=1)
                    coef.append(a)

            return coef

        # Rolling mean for different periods
        self.df['rolling_mean_5'] =  self.df['Close'].rolling(5, closed='both').mean()
        self.df['rolling_mean_10'] = self.df['Close'].rolling(10, closed='both').mean()
        self.df['rolling_mean_21'] = self.df['Close'].rolling(21, closed='both').mean()
        self.df.dropna(inplace=True)

        # Angular coefficients for different periods (trend)
        angular_5 =  get_angular_coef(self.df, 5)
        angular_10 = get_angular_coef(self.df, 10)
        angular_21 = get_angular_coef(self.df, 21)

        self.df['angular_coef_5'] = angular_5
        self.df['angular_coef_10'] = angular_10
        self.df['angular_coef_21'] = angular_21

        # Adjusting dtype
        self.df['target'] = self.df['target'].astype(int)

        return self


    def run_all(self):
        self.normalized_price()
        self.local_max_and_min()
        self.find_new_min()
        self.find_new_max()
        self.rolling_mean()

        return self.df


class ModelTraining():

    def __init__(self, df, model, features, scoring, objective_function, cross_val_iterations=5, optimize_trials=15, scale_data=False):
        self.df = df
        self.model = model
        self.features = features
        self.scoring = scoring
        self.cv_iter = cross_val_iterations
        self.opt_trials = optimize_trials
        self.scale_data = scale_data
        self.objective_function = objective_function


    def benchmark_scaled(self):
        print('Cross Validation')
        def get_cross_validation(model, X, y, scoring='accuracy', n_iter=5):
            results = np.array([])
            for c in range(n_iter):
                kfold = StratifiedKFold()
                cross_results = cross_val_score(model, X, y, scoring=scoring, cv=kfold)
                results = np.concatenate([results, cross_results])

            return results


        scores = np.zeros((1, len(self.features)))
        c = 0
        for feature in self.features:
            scaler = StandardScaler().fit(self.df[feature])
            X_scaled = scaler.transform(self.df[feature])

            values = get_cross_validation(self.model, X_scaled, self.df['target'], scoring=self.scoring, n_iter=self.cv_iter)
            scores[0, c] = values.mean()
            c += 1

        return scores


    def benchmark(self):
        print('Cross Validation')
        def get_cross_validation(model, X, y, scoring='accuracy', n_iter=5):
            results = np.array([])
            for c in range(n_iter):
                kfold = StratifiedKFold()
                cross_results = cross_val_score(model, X, y, scoring=scoring, cv=kfold)
                results = np.concatenate([results, cross_results])

            return results

        scores = np.zeros((1, len(self.features)))
        c = 0
        for feature in self.features:
            X = self.df[feature]

            values = get_cross_validation(self.model, X, self.df['target'], scoring=self.scoring, n_iter=self.cv_iter)
            scores[0, c] = values.mean()
            c += 1

        return scores


    def optimize_scaled(self):
        print('Tuning')
        results = []
        for feature in self.features:
            X = self.df[feature]
            y = self.df['target']
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
            study.optimize(lambda trial: self.objective_function(trial, X=X_scaled, y=y, scoring=self.scoring), n_trials=self.opt_trials)
            results.append(study)

        return results


    def optimize(self):
        print('Tuning')
        results = []
        for feature in self.features:
            X = self.df[feature]
            y = self.df['target']

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
            study.optimize(lambda trial: self.objective_function(trial, X=X, y=y, scoring=self.scoring), n_trials=self.opt_trials)
            results.append(study)

        return results


    def concat_scores(self):  # scaling happens here
        if self.scale_data == True:
            scores = self.benchmark_scaled()

            opt = self.optimize_scaled()
            list_of_results_opt = [opt[c].best_trial.values[0] for c in range(len(opt))]
            opt_array = np.array([list_of_results_opt])

            scores = np.concatenate([scores, opt_array])
            return scores, opt

        else:
            scores = self.benchmark()

            opt = self.optimize()
            list_of_results_opt = [opt[c].best_trial.values[0] for c in range(len(opt))]
            opt_array = np.array([list_of_results_opt])

            scores = np.concatenate([scores, opt_array])
            return scores, opt


    def select_feature(self):
        scores, opt = self.concat_scores()

        no_tune_max = np.where(scores[0] == scores[0].max())[0]
        no_tune_feature = no_tune_max[len(no_tune_max) - 1]

        tune_max = np.where(scores[1] == scores[1].max())[0]
        tune_feature = tune_max[len(tune_max) - 1]

        if scores[1, tune_feature] > scores[0, no_tune_feature]:
            tune = True
            return scores[1, tune_feature], self.features[tune_feature], tune_feature, tune, opt
        else:
            tune = False
            return scores[0, no_tune_feature], self.features[no_tune_feature], no_tune_feature, tune, opt


    def return_model(self):
        print('Fitting')
        if type(self.model).__name__ == 'LogisticRegression':
            from MinMaxClassification.fit_models import fit_LogisticRegression

            score, feature_subset, feature_idx, tune, opt = self.select_feature()
            lr = fit_LogisticRegression(self.df[feature_subset], self.df['target'], tune, opt, feature_idx, self.scale_data)
            print('Model score:', score)
            return lr, feature_subset

        elif type(self.model).__name__ == 'RandomForestClassifier':
            from MinMaxClassification.fit_models import fit_RandomForest

            score, feature_subset, feature_idx, tune, opt = self.select_feature()
            rf = fit_RandomForest(self.df[feature_subset], self.df['target'], tune, opt, feature_idx,
                                        self.scale_data)
            print('Model score:', score)
            return rf, feature_subset

        elif type(self.model).__name__ == 'SVC':
            from MinMaxClassification.fit_models import fit_SVC

            score, feature_subset, feature_idx, tune, opt = self.select_feature()
            svc = fit_SVC(self.df[feature_subset], self.df['target'], tune, opt, feature_idx,
                                        self.scale_data)
            print('Model score:', score)
            return svc, feature_subset



class Simulation():

    def __init__(self, X, y, model, scale_data=False, initial_budget=1000):
        self.model = model
        self.X_fix = X.copy()
        self.X = X
        self.y = y
        self.scale_data = scale_data
        self.initial_budget = initial_budget

        if self.scale_data == True:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)


    def predictions(self):

        preds = np.zeros(len(self.X))
        probs = self.model.predict_proba(self.X)

        idx = 0
        for row in probs:
            if row[0] >= 0.75:
                preds[idx] = 0
            else:
                if row[1] > 0.75:
                    preds[idx] = 1
                else:
                    preds[idx] = 2
            idx += 1

        return preds


    def backtest(self, preds):
        print('Backtesting')
        values = []
        different_from_zero_sells = []
        current_money = self.initial_budget
        n_of_stocks = 0

        previous_buy_price = 0

        for c in range(len(preds)):

            if preds[c] == 0:  # buy

                previous_buy_price = self.X_fix.iloc[c]['Close']
                n_of_stocks_buy = (current_money // self.X_fix.iloc[c]['Close'])
                spent = n_of_stocks_buy * self.X_fix.iloc[c]['Close']

                n_of_stocks += n_of_stocks_buy
                current_money = current_money - spent

            elif preds[c] == 1:  # sell

                earned = n_of_stocks * self.X_fix.iloc[c]['Close']
                current_money = current_money + earned

                if earned != 0:
                    different_from_zero_sells.append((self.X_fix.iloc[c]['Close'] - previous_buy_price) * n_of_stocks)

                if earned > 20000:
                    earned = 0.85 * earned

                n_of_stocks = 0

            elif preds[c] == 2:  # do nothing
                pass

            values.append((n_of_stocks * self.X_fix.iloc[c]['Close']) + current_money)

        money_value = (n_of_stocks * self.X_fix.iloc[len(self.X_fix) - 1]['Close']) + current_money
        return money_value, different_from_zero_sells, values


    def run(self):
        model_predictions = self.predictions()
        money, trades, values = self.backtest(model_predictions)

        return money, trades, values



if __name__ == '__main__':
    pass
