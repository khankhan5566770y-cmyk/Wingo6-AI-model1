import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#,Mock data generate karna
def generate_mock_data(samples=2000):
    X = np.random.randint(0, 10, size=(samples, 5))  # 5 previous results
    y = np.array([1 if np.mean(x) >= 4.5 else 0 for x in X])  # 1 = Big, 0 = Small
    return X, y

# Data banayein
X, y = generate_mock_data()
y_cat = to_categorical(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# Model architecture
model = Sequential([
    Dense(64, input_shape=(5,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 outputs: Small or Big
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Model train karna
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
