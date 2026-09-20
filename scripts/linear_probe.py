import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

X = np.load("data/X_location.npy")
y = np.load("data/y_location.npy")

# re-derive the filenames in the same order used when we built X/y,
# so we can look up *which* image any given row corresponds to later
filenames = []
with open("data/labels.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["location"] != "unknown":
            filenames.append(row["filename"])
filenames = np.array(filenames)

# split into train/test, keeping the outdoor/indoor ratio consistent in both
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, filenames,
    test_size=0.25,      # ~25% held out for testing
    random_state=42,     # fixed seed so the split is reproducible
    stratify=y,
)

print("train size:", len(y_train), Counter(y_train))
print("test size:", len(y_test), Counter(y_test))

# the majority-class baseline, computed on the actual test set we ended up with
baseline_label = Counter(y_train).most_common(1)[0][0]
baseline_acc = (y_test == baseline_label).mean()
print(f"majority-class baseline (always guess '{baseline_label}'): {baseline_acc:.2%}")

# the actual linear probe
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
predictions = probe.predict(X_test)

accuracy = (predictions == y_test).mean()
print(f"linear probe accuracy: {accuracy:.2%}")

# which images did it get wrong?
print("\nmisclassified images:")
for fname, true_label, pred_label in zip(files_test, y_test, predictions):
    if true_label != pred_label:
        print(f"  {fname}: true={true_label}, predicted={pred_label}")