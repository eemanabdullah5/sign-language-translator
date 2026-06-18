import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Hello', 'Thank_You', 'Emergency'])
no_sequences, sequence_length = 30, 30

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(np.array(window).flatten())
        labels.append(label_map[action])

X, y = np.array(sequences), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)
print(f"Training Complete! Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

with open('sign_model.pkv', 'wb') as f:
    pickle.dump(model, f)