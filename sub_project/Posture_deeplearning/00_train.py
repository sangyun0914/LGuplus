from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # Accuracy metrics

import pandas as pd
import pickle 
import os
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
# custom here
PATH_CSV = "Dataset_tf.csv"
SAVE_NAME = "ActionNV2"

# Read Data(CSV)
df = pd.read_csv(PATH_CSV)

# 
X = df.drop('class', axis=1) # features
y = df['class'] # target value

# split to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

print("============= feature =============")
print(X.shape)
print(np.unique(y))
print(y)
print("===================================")
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X.shape[1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, validation_split=0.1,callbacks=[tb_callback])
model.summary()
print("==========================================")
print("# Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)

model.save('ActionNV2.h5')