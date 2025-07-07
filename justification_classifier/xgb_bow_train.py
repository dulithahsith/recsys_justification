import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, make_scorer
import joblib
import os


def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['edu', 'label'])
    df['label'] = df['label'].apply(lambda x: 1 if eval(str(x)) else 0)
    print(f"Unique labels in {path}:", df['label'].unique())
    return df


train_path = os.path.join('justification', 'label_data_train.csv')
dev_path = os.path.join('justification', 'label_data_dev.csv')
test_path = os.path.join('justification', 'label_data_test.csv')

df_train = load_data(train_path)
df_dev = load_data(dev_path)
df_test = load_data(test_path)

vectorizer = CountVectorizer(ngram_range=(1,2), max_features=5000)
X_train = vectorizer.fit_transform(df_train['edu'])
y_train = df_train['label']
X_dev = vectorizer.transform(df_dev['edu'])
y_dev = df_dev['label']
X_test = vectorizer.transform(df_test['edu'])
y_test = df_test['label']

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Use multiple scoring metrics
scoring = {
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=scoring,
    refit='f1',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best F1 score on training (CV):", grid_search.best_score_)

# Find the best index
best_index = grid_search.best_index_

# Get cross-validated recall and precision for the best params
mean_recall = grid_search.cv_results_['mean_test_recall'][best_index]
mean_precision = grid_search.cv_results_['mean_test_precision'][best_index]

# Evaluate on train, dev, and test sets
best_model = grid_search.best_estimator_

# Train set
train_pred = best_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred)
train_report = classification_report(y_train, train_pred)

# Dev set
dev_pred = best_model.predict(X_dev)
dev_acc = accuracy_score(y_dev, dev_pred)
dev_f1 = f1_score(y_dev, dev_pred)
dev_recall = recall_score(y_dev, dev_pred)
dev_precision = precision_score(y_dev, dev_pred)
dev_report = classification_report(y_dev, dev_pred)

# Test set
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_report = classification_report(y_test, test_pred)

print("Train Accuracy:", train_acc)
print("Train F1 Score:", train_f1)
print("Train Recall:", train_recall)
print("Train Precision:", train_precision)
print(train_report)

print("Dev Accuracy:", dev_acc)
print("Dev F1 Score:", dev_f1)
print("Dev Recall:", dev_recall)
print("Dev Precision:", dev_precision)
print(dev_report)

print("Test Accuracy:", test_acc)
print("Test F1 Score:", test_f1)
print("Test Recall:", test_recall)
print("Test Precision:", test_precision)
print(test_report)

with open('xgb_bow_eval.txt', 'w') as f:
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Best F1 (CV): {grid_search.best_score_}\n")
    f.write(f"Best Recall (CV): {mean_recall}\n")
    f.write(f"Best Precision (CV): {mean_precision}\n")
    f.write("\n")
    f.write("=== Train Set ===\n")
    f.write(f"Accuracy: {train_acc}\nF1 Score: {train_f1}\nRecall: {train_recall}\nPrecision: {train_precision}\n")
    f.write(train_report + '\n')
    f.write("=== Dev Set ===\n")
    f.write(f"Accuracy: {dev_acc}\nF1 Score: {dev_f1}\nRecall: {dev_recall}\nPrecision: {dev_precision}\n")
    f.write(dev_report + '\n')
    f.write("=== Test Set ===\n")
    f.write(f"Accuracy: {test_acc}\nF1 Score: {test_f1}\nRecall: {test_recall}\nPrecision: {test_precision}\n")
    f.write(test_report + '\n')

joblib.dump(best_model, 'xgb_bow_model.joblib')
joblib.dump(vectorizer, 'bow_vectorizer.joblib')
