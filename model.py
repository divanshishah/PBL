import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32) / 255.0
    y = np.array(mnist['target'], dtype=np.int32).reshape(-1, 1)
    return X, y

def one_hot_encode(y, num_classes=10):
    y = y.flatten()
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

def backward(X, y, z1, a1, z2, a2, W2):
    m = X.shape[0]
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def train(X, y, input_size=784, hidden_size=128, output_size=10, epochs=20, lr=0.05, batch_size=64):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    m = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            z1, a1, z2, a2 = forward(X_batch, W1, b1, W2, b2)
            loss = compute_loss(y_batch, a2)
            dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, z2, a2, W2)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

def print_sample_weights(W1, W2, num=3):
    print("\nSample weights from Input → Hidden Layer (W1):")
    for i in range(num):
        print(f"Neuron {i}: {W1[:10, i]} ...")
    print("\nSample weights from Hidden → Output Layer (W2):")
    for i in range(num):
        print(f"To Output Neuron {i}: {W2[i, :]}")

X, y = load_mnist()
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = one_hot_encode(y_train_raw)
y_test = one_hot_encode(y_test_raw)

W1, b1, W2, b2 = train(X_train, y_train)
y_pred = predict(X_test, W1, b1, W2, b2)
accuracy = np.mean(y_pred == y_test_raw.flatten())
print(f"\nTest Accuracy: {accuracy:.4f}")
print_sample_weights(W1, W2)
