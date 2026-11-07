"""Reproduce five baseline experiments. Run this in BITS Virtual Lab."""
from pathlib import Path
import json, platform
import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from evaluation import evaluate

ROOT = Path(__file__).resolve().parent
SEED = 271

def main():
    (ROOT/'model').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # sklearn encodes malignant=0. Explicitly reverse for malignant-positive metrics.
    y = (dataset.target == 0).astype(int).rename('target')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.20, stratify=y, random_state=SEED)
    pd.concat([Xtest, ytest], axis=1).to_csv(ROOT/'test_data.csv', index=False, lineterminator='\n')
    models = {
        'Logistic Regression': ('logistic_regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))),
        'Decision Tree': ('decision_tree', DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
        'KNN': ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        'Naive Bayes': ('naive_bayes', make_pipeline(StandardScaler(), GaussianNB())),
        'Support Vector Machine': ('svm', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale'))),
    }
    # Choose the winner using training-only CV, before viewing the test results.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows, test_rows, registry = [], [], {}
    for name, (slug, model) in models.items():
        folds = cross_validate(model, Xtrain, ytrain, cv=cv, scoring={'f1':'f1', 'auc':'roc_auc'}, error_score='raise')
        cv_rows.append({'Model': name, 'CV_F1_mean': folds['test_f1'].mean(), 'CV_F1_std': folds['test_f1'].std(), 'CV_AUC_mean': folds['test_auc'].mean()})
        model.fit(Xtrain, ytrain)
        measured, predictions = evaluate(model, Xtest, ytest)
        test_rows.append({'Model': name, **measured})
        joblib.dump(model, ROOT/'model'/f'{slug}.joblib')
        registry[name] = f'{slug}.joblib'
        (ROOT/'results'/f'{slug}_report.json').write_text(json.dumps({'confusion_matrix':confusion_matrix(ytest,predictions,labels=[0,1]).tolist(), 'classification_report':classification_report(ytest,predictions,labels=[0,1],target_names=['Benign','Malignant'],output_dict=True,zero_division=0)}, indent=2))
    cv_results = pd.DataFrame(cv_rows).sort_values(['CV_F1_mean','CV_AUC_mean'], ascending=False)
    winner = cv_results.iloc[0]['Model']
    cv_results.to_csv(ROOT/'results/cross_validation.csv',index=False,lineterminator='\n')
    metrics = pd.DataFrame(test_rows)
    metrics.to_csv(ROOT/'results/metrics.csv',index=False,lineterminator='\n')
    metadata = {'source':'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic', 'features':X.columns.tolist(),'target':'target','label_mapping':{'0':'Benign','1':'Malignant'},'seed':SEED,'train_rows':len(Xtrain),'test_rows':len(Xtest),'class_counts':y.value_counts().sort_index().to_dict(),'models':registry,'selected_model':winner,'selection_rule':'Maximum mean malignant-class F1 in 5-fold training-only stratified CV; AUC as tie-breaker','python':platform.python_version(),'scikit_learn':sklearn.__version__,'hyperparameters':{name:model.get_params(deep=True).__str__() for name,(_,model) in models.items()}}
    (ROOT/'model/metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False)); print('\nTraining-CV selected model:',winner)
    print('Saved models, held-out test data, CV results, metrics and reports.')
if __name__ == '__main__':
    main()
