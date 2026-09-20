import csv
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

# pick which label to test: run as `python scripts/linear_probe_generic.py light`
label_column = sys.argv[1] if len(sys.argv) > 1 else "location"
print(f"testing label column: '{label_column}'")

X_all = np.load("data/X_all.npy")
filenames_all = np.load("data/filenames_all.npy")

with open("data/labels.csv", newline="") as f:
    label_by_filename = {row["filename"]: row[label_column] for row in csv.DictReader(f)}

# keep only rows where this particular label isn't "unknown"
mask = [label_by_filename[fn] != "unknown" for fn in filenames_all]
X = X_all[mask]
y = np.array([label_by_filename[fn] for fn in filenames_all if label_by_filename[fn] != "unknown"])
filenames = filenames_all[mask]

print("usable images:", len(y), Counter(y))

X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, filenames, test_size=0.25, random_state=42, stratify=y,
)

print("train size:", len(y_train), Counter(y_train))
print("test size:", len(y_test), Counter(y_test))

baseline_label = Counter(y_train).most_common(1)[0][0]
baseline_acc = (y_test == baseline_label).mean()
print(f"majority-class baseline (always guess '{baseline_label}'): {baseline_acc:.2%}")

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
predictions = probe.predict(X_test)

accuracy = (predictions == y_test).mean()
print(f"linear probe accuracy: {accuracy:.2%}")

print("\nmisclassified images:")
for fname, true_label, pred_label in zip(files_test, y_test, predictions):
    if true_label != pred_label:
        print(f"  {fname}: true={true_label}, predicted={pred_label}")