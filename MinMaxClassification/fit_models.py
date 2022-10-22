
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def return_fit(X, y, clf, scaled):
    if scaled == True:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        clf.fit(X_scaled, y)
    else:
        clf.fit(X, y)
    return clf



def fit_LogisticRegression(X, y, tune, opt, feature_idx, scaled=True):

    if tune == True:
        best_opt = opt[feature_idx].best_trial.params
        clf = LogisticRegression(C=best_opt['C'], max_iter=500,
                                class_weight='balanced')
    else:
        clf = LogisticRegression(max_iter=500, class_weight='balanced')

    clf = return_fit(X, y, clf, scaled)
    return clf


def fit_RandomForest(X, y, tune, opt, feature_idx, scaled=False):
    if tune == True:
        best_opt = opt[feature_idx].best_trial.params
        clf = RandomForestClassifier(
            n_estimators=best_opt['n_estimators'],
            criterion=best_opt['criterion'],
            max_depth=best_opt['max_depth'],
            max_features=best_opt['max_features'],
            min_samples_split=best_opt['min_samples_split'],
            class_weight='balanced')
    else:
        clf = RandomForestClassifier(class_weight='balanced')

    clf = return_fit(X, y, clf, scaled)
    return clf


def fit_SVC(X, y, tune, opt, feature_idx, scaled=True):
    if tune == True:
        best_opt = opt[feature_idx].best_trial.params
        if best_opt['kernel'] == 'poly':
            clf = SVC(C=best_opt['C'], kernel=best_opt['kernel'], gamma=best_opt['gamma'], degree=best_opt['degree'],
                      class_weight='balanced', probability=True)
        else:
            clf = SVC(C=best_opt['C'], kernel=best_opt['kernel'], gamma=best_opt['gamma'], coef0=best_opt['coef0'],
                      class_weight='balanced', probability=True)
    else:
        clf = SVC(class_weight='balanced', probability=True)

    clf = return_fit(X, y, clf, scaled)
    return clf


if __name__ == '__main__':
    pass