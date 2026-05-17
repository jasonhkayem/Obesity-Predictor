"""
src/train.py
Full training pipeline: load → preprocess → baseline → compare models → tune → save.

Run from the project root:
    python src/train.py
"""
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

np.random.seed(42)

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, '..', 'data', 'Obesity Data Set.csv')
MODEL_PATH = os.path.join(_HERE, '..', 'models', 'pipeline.pkl')

# columns that go through MinMaxScaler
NUMERICAL_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# all continuous columns that go through MinMaxScaler — raw numerics + BMI + engineered features
ALL_SCALE_COLS = NUMERICAL_COLS + ['BMI', 'meal_regularity_score', 'hydration_activity_ratio', 'tech_activity_ratio']

# CAEC and CALC have a natural severity ordering (none → always), so ordinal encoding
# preserves that relationship more compactly than one-hot; OHE would imply no ordering
CAEC_CALC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

# fixed category list so train and inference always produce the same dummy columns,
# regardless of which MTRANS values happen to be present in a given batch
MTRANS_CATS = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']

# adjacent obesity/overweight grades share overlapping behavioral signatures;
# collapsing them reduces label noise and focuses each class on a clinically distinct band
CLASS_MERGE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4}
LABEL_MAP   = {0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight', 4: 'Obesity'}

TARGET_RAW_MAP = {
    'Insufficient_Weight': 0, 'Normal_Weight': 1,
    'Overweight_Level_I': 2,  'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,      'Obesity_Type_II': 5,     'Obesity_Type_III': 6,
}


# Preprocessing

class ObesityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Single-step transformer encapsulating the full notebook pipeline.

    Putting imputation, encoding, BMI, scaling, feature engineering, and feature
    selection inside fit()/transform() means sklearn Pipeline calls fit() only on
    X_train — so scalers and feature selectors never see test-set data.
    That is the leakage fix: the notebook fit its MinMaxScaler on the full dataset
    before splitting, inflating the apparent accuracy of distance-sensitive models.
    """

    def fit(self, X, y=None):
        df = X.copy()

        # Imputation
        self.age_knn_    = KNNImputer(n_neighbors=5).fit(df[['Age']])
        self.height_knn_ = KNNImputer(n_neighbors=5).fit(df[['Height']])
        self.weight_knn_ = KNNImputer(n_neighbors=5).fit(df[['Weight']])

        # mode for low-missing-rate binary/ordinal/categorical columns; stable enough
        # that the most frequent value is a safe default when missingness is < 5%
        self.modes_ = {c: df[c].mode()[0]
                       for c in ['family_history_with_overweight', 'FAVC', 'CAEC',
                                 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']}

        # NCP: notebook chose median because 3 meals/day is the norm;
        # median is more robust than mean for this integer-valued column
        self.ncp_med_ = df['NCP'].median()

        # random imputation pools for Gender and FCVC: sampling from the observed
        # marginal distribution avoids amplifying the majority category the way mode would
        self.gender_pool_ = df['Gender'].dropna().values
        self.fcvc_pool_   = df['FCVC'].dropna().values

        # 2. Apply transforms 
        rng = np.random.RandomState(42)
        df  = self._impute(df, rng)
        df  = self._encode(df)

        df = self._engineer(df)

        self.num_scaler_ = MinMaxScaler().fit(df[ALL_SCALE_COLS])
        df[ALL_SCALE_COLS] = self.num_scaler_.transform(df[ALL_SCALE_COLS])

        self.feature_names_ = list(df.columns)
        return self

    def transform(self, X):
        df  = X.copy()
        rng = np.random.RandomState(42)
        df  = self._impute(df, rng)
        df  = self._encode(df)
        df  = self._engineer(df)
        df[ALL_SCALE_COLS] = self.num_scaler_.transform(df[ALL_SCALE_COLS])
        return df.values

    # Private helpers

    def _impute(self, df, rng):
        df = df.copy()
        # some CSV rows store FAF/TUE as strings due to data-entry issues; coerce first
        df['FAF'] = pd.to_numeric(df['FAF'], errors='coerce')
        df['TUE'] = pd.to_numeric(df['TUE'], errors='coerce')

        df['Age']    = self.age_knn_.transform(df[['Age']]).ravel()
        df['Height'] = self.height_knn_.transform(df[['Height']]).ravel()
        df['Weight'] = self.weight_knn_.transform(df[['Weight']]).ravel()

        df['Gender'] = df['Gender'].apply(
            lambda x: rng.choice(self.gender_pool_) if pd.isna(x) else x)
        df['FCVC'] = df['FCVC'].apply(
            lambda x: rng.choice(self.fcvc_pool_) if pd.isna(x) else x)

        df['NCP'] = df['NCP'].fillna(self.ncp_med_)
        for col, val in self.modes_.items():
            df[col] = df[col].fillna(val)
        return df

    def _encode(self, df):
        df = df.copy()
        df['Gender']                         = df['Gender'].map({'Male': 0, 'Female': 1})
        df['family_history_with_overweight']  = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
        df['FAVC']  = df['FAVC'].map({'yes': 1, 'no': 0})
        df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})
        df['SCC']   = df['SCC'].map({'yes': 1, 'no': 0})
        df['CAEC']  = df['CAEC'].map(CAEC_CALC_MAP)
        df['CALC']  = df['CALC'].map(CAEC_CALC_MAP)
        # fixed-list OHE: guarantees identical columns on train and at inference time,
        # regardless of which MTRANS values appear in the current batch
        for cat in MTRANS_CATS:
            df[f'MTRANS_{cat}'] = (df['MTRANS'] == cat).astype(int)
        df = df.drop(columns=['MTRANS'])
        return df

    def _engineer(self, df):
        df = df.copy()
        # domain-motivated interaction terms: eating regularity, hydration vs. activity,
        # sedentary behaviour vs. activity, and the compound genetic+diet risk
        df['BMI']                      = df['Weight'] / (df['Height'] ** 2)
        df['meal_regularity_score']    = df['FCVC'] * df['NCP'] - df['CAEC']
        df['hydration_activity_ratio'] = df['CH2O'] / (df['FAF'] + 0.1)  # +0.1 prevents /0
        df['tech_activity_ratio']      = df['TUE']  / (df['FAF'] + 0.1)
        # multiplicative: both genetic history AND junk food must be present to produce risk;
        # additive would overstate the effect when only one factor is present
        df['genetic_diet_risk']        = df['family_history_with_overweight'] * df['FAVC']
        return df


# Utilities 

def make_pipeline(model):
    """Return a fresh Pipeline with an independent preprocessor instance.

    Using a new ObesityPreprocessor() each time avoids shared fitted state
    across the model-comparison loop.
    """
    return Pipeline([('preprocessor', ObesityPreprocessor()), ('model', model)])


def plot_confusion_matrix(y_test, y_pred, model_name):
    labels_sorted   = sorted(LABEL_MAP.keys())
    display_labels  = [LABEL_MAP[k] for k in labels_sorted]
    cm   = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # macro-F1 is more informative than accuracy alone when classes are balanced (~25% each):
    # it averages F1 per class equally, so a model that fails on one class cannot hide
    # behind the majority. Accuracy would also be fine here, but macro-F1 is the stricter check.
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"  {name:35s}  acc={acc:.4f}  macro-F1={f1:.4f}")
    plot_confusion_matrix(y_test, y_pred, name)
    return acc, f1


# Main
def main():
    # Load
    df = pd.read_csv(DATA_PATH)

    # Encode target 
    df['NObeyesdad'] = df['NObeyesdad'].map(TARGET_RAW_MAP)
    df = df.dropna(subset=['NObeyesdad'])       # drop ≈5% rows with unrecognised target
    df['NObeyesdad'] = df['NObeyesdad'].astype(int).map(CLASS_MERGE)

    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"Class distribution (train):\n{y_train.value_counts().sort_index().to_string()}\n")

    # Baseline
    # DummyClassifier establishes the floor: any real model must beat "always guess
    # the most frequent class". This sanity-checks that our preprocessing isn't broken.
    print("=== Baseline ===")
    evaluate('Dummy (most-frequent)',
             make_pipeline(DummyClassifier(strategy='most_frequent')),
             X_train, X_test, y_train, y_test)

    # Model comparison
    print("\n=== Model comparison (default hyperparameters) ===")
    models = [
        # lbfgs handles multinomial loss natively; max_iter=1000 because the default
        # 100 iterations frequently fails to converge on this many features
        ('Logistic Regression',
         LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)),

        # KNN is sensitive to feature scale — the upstream MinMaxScaler makes distances fair
        ('K-Nearest Neighbors', KNeighborsClassifier()),

        # SVM: strong margin-based baseline; feasible at 2000 rows
        ('Support Vector Machine', SVC()),

        ('Decision Tree', DecisionTreeClassifier(random_state=42)),

        # Random Forest: bagging reduces variance, feature importances
        # enable explainability, and it is not sensitive to feature scale (unlike KNN/SVM)
        # — so it generalises well even when some features remain on different scales
        ('Random Forest', RandomForestClassifier(random_state=42)),

        ('Naive Bayes', GaussianNB()),
    ]

    results = {}
    for name, model in models:
        acc, f1 = evaluate(name, make_pipeline(model), X_train, X_test, y_train, y_test)
        results[name] = {'acc': acc, 'f1': f1}

    plt.close('all')

    # Hyperparameter tuning on Random Forest
    print("\n=== RandomizedSearchCV on Random Forest ===")
    param_dist = {
        # n_estimators: more trees lower variance; returns diminish above ~300
        'model__n_estimators':      [100, 200, 300, 500],
        # max_depth=None grows fully; shallow depths act as regularisation against overfitting
        'model__max_depth':         [None, 10, 20, 30],
        # min_samples_split: larger values prevent splits on tiny, noisy leaf nodes
        'model__min_samples_split': [2, 5, 10],
        'model__max_features':      ['sqrt', 'log2'],
    }
    search = RandomizedSearchCV(
        make_pipeline(RandomForestClassifier(random_state=42)),
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")

    best_pipeline = search.best_estimator_
    y_pred      = best_pipeline.predict(X_test)
    tuned_acc   = accuracy_score(y_test, y_pred)
    tuned_f1    = f1_score(y_test, y_pred, average='macro')
    print(f"\n  {'Random Forest (tuned)':35s}  acc={tuned_acc:.4f}  macro-F1={tuned_f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, 'Random Forest (tuned)')

    # Feature importances
    rf         = best_pipeline.named_steps['model']
    pre        = best_pipeline.named_steps['preprocessor']
    importances = pd.Series(rf.feature_importances_, index=pre.feature_names_)
    print(f"\nFeature importances (descending):")
    print(importances.sort_values(ascending=False).round(4).to_string())

    # Save
    # Single file bundles the fitted preprocessor (scalers, imputers, selected features)
    # and the model. app.py only needs:
    #   pipeline = joblib.load('models/pipeline.pkl')
    #   pipeline.predict(raw_df)
    # — no separate scaler or feature-list file required.
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nPipeline saved → {MODEL_PATH}")


if __name__ == '__main__':
    main()
