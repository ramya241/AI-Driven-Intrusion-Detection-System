import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATA
df = pd.read_csv("02-16-2018.csv", low_memory=False)

print("Original Shape:", df.shape)

# CLEAN DATA
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

df.drop_duplicates(inplace=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("After Cleaning:", df.shape)

# REMOVE RARE CLASSES
min_samples = 50
valid_classes = df["Label"].value_counts()
valid_classes = valid_classes[valid_classes >= min_samples].index
df = df[df["Label"].isin(valid_classes)]


# BINARY LABEL

df["Label"] = df["Label"].apply(
    lambda x: "Benign" if x == "Benign" else "Attack"
)

print("\nClass Distribution:")
print(df["Label"].value_counts())

# FEATURES / LABEL
X = df.drop("Label", axis=1)
y = df["Label"]

#REMOVE LEAKAGE FEATURES (VERY IMPORTANT)
leak_cols = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s'
]
X.drop(columns=[col for col in leak_cols if col in X.columns], inplace=True)

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# RANDOM SPLIT (BETTER GENERALIZATION)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# MODELS (REGULARIZED)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,             # 🔥 reduced depth
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1
)

# TRAIN
print("\n Training models...")
rf.fit(X_train, y_train)
et.fit(X_train, y_train)


# CROSS VALIDATION (ANTI-OVERFITTING CHECK)

print("\n Cross Validation Scores:")
rf_cv = cross_val_score(rf, X_train, y_train, cv=5)
et_cv = cross_val_score(et, X_train, y_train, cv=5)

print("RF CV Mean:", rf_cv.mean())
print("ET CV Mean:", et_cv.mean())


# EVALUATION

def evaluate(name, model):
    y_pred = model.predict(X_test)

    print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

evaluate("RandomForest", rf)
evaluate("ExtraTrees", et)


# SAVE MODEL

best_model = rf

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\n Model saved successfully!")