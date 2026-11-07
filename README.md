# CellScope Breast Cancer classification

This project compares five classification models on the Breast Cancer Wisconsin (Diagnostic) dataset. You can try them out in a Streamlit app that only makes predictions and never retrains the models.

## a. Problem statement
This project compares five supervised classification algorithms on a single task: predicting whether a breast mass is malignant or benign from numeric measurements of its cell nuclei. All five models are tested on the same held-out test set and scored on six metrics. There is also an interactive Streamlit app where you can upload a test CSV and see how the saved models perform on it. The app doesn’t retrain anything. This is a learning exercise only and is not meant for clinical diagnosis.

## b. Dataset description
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic).
- **Source:** https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- **Loader:** scikit-learn `load_breast_cancer(as_frame=True)`, a bundled copy of the UCI dataset.
- **Size:** 569 instances, 30 numeric input features (meets the minimum of 500 instances and 12 features).
- **Features:** measurements of cell nuclei from digitized fine needle aspirate (FNA) images. Ten measurement families (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each recorded as mean, standard error and worst value.
- **Target:** binary. 0 = benign (357), 1 = malignant (212). The loader's original encoding (malignant = 0) is reversed so that malignant is the positive class.
- **Missing values:** none in the dataset. The app rejects uploads with missing, infinite or non-numeric values.
- **Split:** stratified 80/20 split, `random_state=271`. 455 training rows and 114 test rows (72 benign, 42 malignant). The test rows are saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside a pipeline for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree uses unscaled inputs. All transformations are fitted on training rows only.
- **Model selection:** 5-fold shuffled stratified cross-validation on the training rows only, ranked by mean malignant-class F1 with mean AUC as tie-breaker. The test set is used only for the final comparison.
- **Limitations:** small historical dataset, a single held-out split, strongly correlated measurements, no external clinical validation.

## c. GitHub Repository Link
GitHub repository: **https://github.com/PreetiYadav2801/cellscope-breast-cancer-classifier.git**

Live Streamlit app: **https://cellscope-breast-cancer-classifier-ehzaygcet6hq9xsvv6cwcu.streamlit.app/**

Repository contents:
```
ml_assignment/
|-- app.py             Streamlit frontend (inference only)
|-- train_models.py    training, CV, evaluation, model saving
|-- evaluation.py      shared six-metric evaluation function
|-- requirements.txt   pinned runtime dependencies
|-- README.md
|-- test_data.csv      114 held-out rows: 30 features + target
|-- model/             5 saved .joblib models + metadata.json
|-- results/           metrics, CV table, per-model reports
```

## d. Models used
1. **Logistic Regression:** StandardScaler, C=1.0, lbfgs solver, max_iter=3000, random_state=271.
2. **Decision Tree Classifier:** gini criterion, max_depth=5, min_samples_leaf=3, random_state=271.
3. **K-Nearest Neighbor Classifier:** StandardScaler, n_neighbors=7, uniform weights, Euclidean distance (Minkowski p=2).
4. **Naive Bayes Classifier (Gaussian):** StandardScaler, GaussianNB with var_smoothing=1e-9.
5. **Support Vector Machine:** StandardScaler, RBF kernel, C=1.0, gamma='scale'. AUC uses `decision_function` scores.

No hyperparameter search was performed. Other parameters use scikit-learn defaults. `model/metadata.json` records the full settings.

### Comparison table (held-out test set, 114 rows)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9912 | 0.9917 | 1.0000 | 0.9762 | 0.9880 | 0.9812 |
| Decision Tree | 0.8947 | 0.9491 | 0.9412 | 0.7619 | 0.8421 | 0.7741 |
| KNN | 0.9737 | 0.9828 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Naive Bayes | 0.9298 | 0.9884 | 0.9722 | 0.8333 | 0.8974 | 0.8504 |
| Support Vector Machine (SVM) | 0.9825 | 0.9917 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

Precision, recall and F1 treat malignant (1) as the positive class, with zero_division=0. AUC uses the positive-class probability, or the SVM decision score. The app shows AUC as N/A when an uploaded file contains only one class.

### Observations on model performance
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | The best results on the held-out test set: accuracy 0.9912, F1 0.9880 and MCC 0.9812. It caught 41 of the 42 malignant cases and never labelled a benign case as malignant (precision 1.0). Once the features are standardised, the two classes are almost linearly separable, so a simple linear model generalises very well. |
| Decision Tree | The weakest model (accuracy 0.8947, recall 0.7619). It missed 10 of the 42 malignant cases and wrongly flagged 2 benign ones. Because its depth is limited, the tree is easy to read and explain. The catch is that it can only split on one feature at a time, so it traces the boundary between classes less accurately than the other models. |
| KNN | Performs well once the features are standardised (accuracy 0.9737, F1 0.9639). It missed 2 malignant cases and produced 1 false positive. Its results depend heavily on proper feature scaling and on the chosen number of neighbours (k = 7). |
| Naive Bayes | It ranks cases very well (AUC 0.9884), but at the default threshold its recall (0.8333) and F1 (0.8974) drop noticeably. It missed 7 malignant cases and gave 1 false positive. Many of the features are strongly correlated with each other, which breaks the model’s independence assumption and makes its probability estimates overconfident. |
| Support Vector Machine (SVM) | A close second (accuracy 0.9825, F1 0.9762), with one malignant and one benign case misclassified. Its AUC (0.9917) matches Logistic Regression’s, so the two models rank cases equally well. SVM simply made one more mistake at the decision threshold. |
| **Overall winner** | **Logistic Regression.** It came first in cross-validation on the training data (mean F1 0.9676, SD 0.0216), and it was chosen on that basis before the test set was ever used. It then also scored best on the held-out test set (F1 0.9880). SVM came second both times. |

### Training-only cross-validation (5-fold, stratified)
| Model | CV F1 mean | CV F1 std | CV AUC mean |
| --- | --- | --- | --- |
| Logistic Regression | 0.9676 | 0.0216 | 0.9961 |
| Support Vector Machine | 0.9488 | 0.0208 | 0.9948 |
| KNN | 0.9416 | 0.0315 | 0.9831 |
| Decision Tree | 0.9166 | 0.0291 | 0.9486 |
| Naive Bayes | 0.9062 | 0.0264 | 0.9875 |

## Streamlit app features
- **CSV upload** of test data (all 30 feature columns plus `target`). The schema is checked and the training feature order is restored.
- **Model selection dropdown** with all five classifiers.
- **Evaluation metrics:** six metric cards for the selected model, plus a comparison table of all five models on the uploaded data (downloadable as CSV).
- **Confusion matrix and classification report** for the selected model.
- Download button for the supplied `test_data.csv`.
- Saved models are only loaded and used for prediction. Uploaded data is never used for fitting.

## How to run
Tested with Python 3.13 and the versions in `requirements.txt`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py        # optional: retrains and regenerates model/, results/, test_data.csv
streamlit run app.py
```

Then upload `test_data.csv` from the sidebar.

## Deployment (Streamlit Community Cloud)
1. Push this folder to a GitHub repository, including `model/` and `test_data.csv`.
2. On https://share.streamlit.io choose **Create app**, select the repository, the branch (`main`) and `app.py`.
3. Under **Advanced settings** select Python 3.13, then deploy.
4. Open the app link, upload `test_data.csv` and check each of the five models.
