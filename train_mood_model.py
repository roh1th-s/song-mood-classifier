import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'features_30_sec.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

df = pd.read_csv(DATA_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

selected_moods = ['angry', 'sad', 'happy', 'romantic']
df = df[df['label'].isin(selected_moods)].reset_index(drop=True)

print(f"Dataset shape  : {df.shape}")
print(f"Class counts   :\n{df['label'].value_counts()}")

X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

X = X.copy()
X['energy_tempo_ratio']      = X['rms_mean'] / (X['chroma_stft_mean'] + 1e-6)
X['spectral_energy']         = X['spectral_centroid_mean'] * X['rms_mean']
X['tonal_brightness']        = X['chroma_stft_mean'] * X['spectral_centroid_mean']
X['chroma_variance_ratio']   = X['chroma_stft_var'] / (X['chroma_stft_mean'] + 1e-6)
X['brightness_energy_ratio'] = X['spectral_centroid_mean'] / (X['rms_mean'] + 1e-6)
X['bandwidth_stability']     = X['spectral_bandwidth_mean'] / (X['spectral_bandwidth_var'] + 1e-6)
X['mel_energy_ratio']        = X['melspectrogram_mean'] / (X['melspectrogram_var'] + 1e-6)
X['harmonic_richness']       = X['chroma_stft_mean'] * X['spectral_bandwidth_mean']

print(f"\nFeature count  : {X.shape[1]}")
print(f"Random baseline: {1/y.nunique():.2%}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X = X.drop(columns=['fourier_tempogram_mean', 'fourier_tempogram_var'], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain size : {X_train_scaled.shape[0]}")
print(f"Test size  : {X_test_scaled.shape[0]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_models = {
    "Logistic Regression": LogisticRegression(
        solver='lbfgs', max_iter=2000, C=1.0, random_state=42),
    "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
    "SVM (RBF)":           SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
}
print("BASELINE MODEL COMPARISON")
results = {}
for name, model in baseline_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    model.fit(X_train_scaled, y_train)
    y_pred   = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {'model': model, 'y_pred': y_pred, 'test_acc': test_acc}

    print(f"\n{name}")
    print(f"  CV F1    : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Test Acc : {test_acc:.3f}")

print("HYPERPARAMETER TUNING (GridSearchCV)")
print("\nTuning Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42),
    {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0
)
lr_grid.fit(X_train_scaled, y_train)
y_pred_lr = lr_grid.best_estimator_.predict(X_test_scaled)
print(f"  Best params : {lr_grid.best_params_}")
print(f"  Best CV F1  : {lr_grid.best_score_:.3f}")
print(f"  Test Acc    : {accuracy_score(y_test, y_pred_lr):.3f}")
results["LR (Tuned)"] = {'model': lr_grid.best_estimator_, 'y_pred': y_pred_lr,
                          'test_acc': accuracy_score(y_test, y_pred_lr)}

print("\nTuning KNN...")
knn_grid = GridSearchCV(
    KNeighborsClassifier(weights='distance'),
    {'n_neighbors': [3, 5, 7, 9, 11, 15]},
    cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0
)
knn_grid.fit(X_train_scaled, y_train)
y_pred_knn = knn_grid.best_estimator_.predict(X_test_scaled)
print(f"  Best params : {knn_grid.best_params_}")
print(f"  Best CV F1  : {knn_grid.best_score_:.3f}")
print(f"  Test Acc    : {accuracy_score(y_test, y_pred_knn):.3f}")
results["KNN (Tuned)"] = {'model': knn_grid.best_estimator_, 'y_pred': y_pred_knn,
                           'test_acc': accuracy_score(y_test, y_pred_knn)}

print("\nTuning SVM...")
svm_grid = GridSearchCV(
    SVC(random_state=42),
    {
        'C':      [0.1, 1, 10, 100],
        'gamma':  ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly']
    },
    cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0
)
svm_grid.fit(X_train_scaled, y_train)
y_pred_svm = svm_grid.best_estimator_.predict(X_test_scaled)
print(f"  Best params : {svm_grid.best_params_}")
print(f"  Best CV F1  : {svm_grid.best_score_:.3f}")
print(f"  Test Acc    : {accuracy_score(y_test, y_pred_svm):.3f}")
results["SVM (Tuned)"] = {'model': svm_grid.best_estimator_, 'y_pred': y_pred_svm,
                           'test_acc': accuracy_score(y_test, y_pred_svm)}

print("VOTING CLASSIFIER (SVM + LR + KNN)")

voting = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
        ('lr',  LogisticRegression(solver='lbfgs',
                                   max_iter=2000, C=1.0, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=7, metric='euclidean', weights='distance')),
    ],
    voting='soft'
)
voting.fit(X_train_scaled, y_train)
y_pred_vote = voting.predict(X_test_scaled)
vote_acc    = accuracy_score(y_test, y_pred_vote)
vote_cv     = cross_val_score(voting, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')

print(f"\nVoting Classifier")
print(f"  CV F1    : {vote_cv.mean():.3f} ± {vote_cv.std():.3f}")
print(f"  Test Acc : {vote_acc:.3f}")
results["Voting (SVM+LR+KNN)"] = {'model': voting, 'y_pred': y_pred_vote, 'test_acc': vote_acc}

print("FINAL RESULTS SUMMARY")
for name, res in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"  {name:35s} → {res['test_acc']:.3f}")

best_name = max(results, key=lambda k: results[k]['test_acc'])
best      = results[best_name]

print(f"\nBest model : {best_name}")
print(f"Test Acc   : {best['test_acc']:.3f}")
print("\n" + classification_report(y_test, best['y_pred'], target_names=le.classes_))

cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.title(f'Confusion Matrix — {best_name}\nAccuracy: {best["test_acc"]:.2%}')
plt.show()

joblib.dump(best['model'],        os.path.join(OUTPUT_DIR, 'best_model.pkl'))
joblib.dump(scaler,               os.path.join(OUTPUT_DIR, 'scaler.pkl'))
joblib.dump(le,                   os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
joblib.dump(X.columns.tolist(),   os.path.join(OUTPUT_DIR, 'feature_columns.pkl'))
print(f"\nModel artifacts saved to: {OUTPUT_DIR}")