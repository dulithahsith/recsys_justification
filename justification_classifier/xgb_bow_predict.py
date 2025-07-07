import pandas as pd
import joblib
import os

clf = joblib.load('xgb_bow_model.joblib')
vectorizer = joblib.load('bow_vectorizer.joblib')

test_path = os.path.join('justification', 'label_data_test.csv')
df_test = pd.read_csv(test_path, sep='\t', header=None, names=['edu', 'label'])
X_test = vectorizer.transform(df_test['edu'])

y_pred = clf.predict(X_test)
df_test['predicted_label'] = y_pred
df_test['predicted_label'] = df_test['predicted_label'].map(
    {1: 'True', 0: 'False'})

df_test.to_csv('xgb_bow_predictions.csv', index=False)
print("Predictions saved to xgb_bow_predictions.csv")


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
