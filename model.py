import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Generate mock data
def generate_mock_data(samples=1000):
    X = np.random.randint(0, 10, size=(samples, 5))  # 5 recent numbers
    y = np.array([1 if np.mean(x) >= 4.5 else 0 for x in X])  # 1=Big, 0=Small
    return X, y

X, y = generate_mock_data(2000)
y_cat = to_categorical(y)

Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# Neural Network
model = Sequential([
    Dense(16, input_dim=5, activation='relu'),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: Big or Small
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# Save model
model.save("big_small_predictor.h5")
print("✅ Model trained & saved as big_small_predictor.h5")
