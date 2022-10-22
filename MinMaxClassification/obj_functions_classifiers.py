
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def logisticRegression_objective_function(trial, X=None, y=None, scoring='accuracy'):
    # LR optimization

    C = trial.suggest_float('C', 0.01, 100)
    classifier_obj = LogisticRegression(C=C, class_weight='balanced', max_iter=500)

    score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=StratifiedKFold(n_splits=5), scoring=scoring)
    acc = score.mean()
    return acc


def SVC_objective_function(trial, X=None, y=None, scoring='accuracy'):
    C = trial.suggest_float('C', 0.01, 100)
    kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 0.01, 20)

    classifier_obj = SVC(class_weight='balanced', probability=True)
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 4)
        classifier_obj = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, class_weight='balanced', probability=True)
    else:
        coef0 = trial.suggest_float('coef0', 0.01, 10)
        classifier_obj = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, class_weight='balanced', probability=True)

    score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=StratifiedKFold(n_splits=5), scoring=scoring)
    acc = score.mean()
    return acc


def randomForest_objective_function(trial, X=None, y=None, scoring='accuracy'):
    # RF optimization

    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_features = trial.suggest_float('max_features', 0.1, 0.8)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 0.8)

    classifier_obj = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        class_weight='balanced')

    score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=StratifiedKFold(n_splits=5), scoring=scoring)
    acc = score.mean()
    return acc


if __name__ == '__main__':
    pass