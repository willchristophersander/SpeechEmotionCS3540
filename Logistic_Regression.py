import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report


#Logistic Regression Set-up

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(X, Y_onehot, Theta):
    m = X.shape[0]
    logits = X @ Theta
    probs = softmax(logits)
    cost = -np.sum(Y_onehot * np.log(probs + 1e-15)) / m  # add epsilon to avoid log(0)
    return cost

def gradient_descent(X, Y_onehot, Theta, lr, iters):
    m = X.shape[0]
    cost_history = []
    
    for i in range(iters):
        logits = X @ Theta             
        probs = softmax(logits)
        gradient = (1/m) * X.T @ (probs - Y_onehot)
        Theta -= lr * gradient         
        cost_history.append(compute_cost(X, Y_onehot, Theta))
    
    return Theta, cost_history

# Read in 100+ features 

data = pd.read_csv("C:/Users/sashi/Downloads/crema_d_features_cleaned.csv")

data['ones'] = 1

data = data.dropna()

emotion_map = {
    'Anger': 0,
    'Happy': 1,
    'Sad': 2,
    'Neutral': 3,
    'Disgust': 4,
    'Fear': 5
}

data['Emotion_num'] = data['emotion'].map(emotion_map)

print(data[['emotion', 'Emotion_num']].head(n=10))

X = data[['ones',"mfcc_1_mean","mfcc_1_std","mfcc_1_min","mfcc_1_max","mfcc_1_skew","mfcc_1_kurtosis",
"mfcc_2_mean","mfcc_2_std","mfcc_2_min","mfcc_2_max","mfcc_2_skew","mfcc_2_kurtosis",
"mfcc_3_mean","mfcc_3_std","mfcc_3_min","mfcc_3_max","mfcc_3_skew","mfcc_3_kurtosis",
"mfcc_4_mean","mfcc_4_std","mfcc_4_min","mfcc_4_max","mfcc_4_skew","mfcc_4_kurtosis",
"mfcc_5_mean","mfcc_5_std","mfcc_5_min","mfcc_5_max","mfcc_5_skew","mfcc_5_kurtosis",
"mfcc_6_mean","mfcc_6_std","mfcc_6_min","mfcc_6_max","mfcc_6_skew","mfcc_6_kurtosis",
"mfcc_7_mean","mfcc_7_std","mfcc_7_min","mfcc_7_max","mfcc_7_skew","mfcc_7_kurtosis",
"mfcc_8_mean","mfcc_8_std","mfcc_8_min","mfcc_8_max","mfcc_8_skew","mfcc_8_kurtosis",
"mfcc_9_mean","mfcc_9_std","mfcc_9_min","mfcc_9_max","mfcc_9_skew","mfcc_9_kurtosis",
"mfcc_10_mean","mfcc_10_std","mfcc_10_min","mfcc_10_max","mfcc_10_skew","mfcc_10_kurtosis",
"mfcc_11_mean","mfcc_11_std","mfcc_11_min","mfcc_11_max","mfcc_11_skew","mfcc_11_kurtosis",
"mfcc_12_mean","mfcc_12_std","mfcc_12_min","mfcc_12_max","mfcc_12_skew","mfcc_12_kurtosis",
"mfcc_13_mean","mfcc_13_std","mfcc_13_min","mfcc_13_max","mfcc_13_skew","mfcc_13_kurtosis",
"spectral_centroid_mean","spectral_centroid_std","spectral_centroid_skew",
"spectral_rolloff_mean","spectral_rolloff_std",
"spectral_bandwidth_mean","spectral_bandwidth_std",
"zcr_mean","zcr_std","zcr_skew",
"chroma_1_mean","chroma_1_std","chroma_2_mean","chroma_2_std","chroma_3_mean","chroma_3_std",
"chroma_4_mean","chroma_4_std","chroma_5_mean","chroma_5_std","chroma_6_mean","chroma_6_std",
"chroma_7_mean","chroma_7_std","chroma_8_mean","chroma_8_std","chroma_9_mean","chroma_9_std",
"chroma_10_mean","chroma_10_std","chroma_11_mean","chroma_11_std","chroma_12_mean","chroma_12_std",
"tonnetz_1_mean","tonnetz_1_std","tonnetz_2_mean","tonnetz_2_std","tonnetz_3_mean","tonnetz_3_std",
"tonnetz_4_mean","tonnetz_4_std","tonnetz_5_mean","tonnetz_5_std","tonnetz_6_mean","tonnetz_6_std",
"tempo","onset_count","onset_rate",
"rms_mean","rms_std","rms_skew",
"energy_entropy_mean","energy_entropy_std",
"harmonic_centroid_mean","harmonic_centroid_std",
"percussive_centroid_mean","percussive_centroid_std",
"harmonic_to_noise_ratio",
"f0_mean","f0_std","f0_min","f0_max","f0_skew","f0_kurtosis","f0_range",
"voicing_ratio","duration"
]].to_numpy()
Y = data[['Emotion_num']].to_numpy().reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

m, n = X.shape
K = len(np.unique(Y))

Y_onehot = np.zeros((m, K))
for i in range(m):
    Y_onehot[i, Y[i]] = 1

print(X.shape, Y.shape)

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print("Reduced features:", X_reduced.shape[1])



# Train and Test model

X_train, X_test, Y_train, Y_test = train_test_split(
    X_reduced, Y_onehot, test_size=0.2, random_state=42
)

n_features = X_train.shape[1]
n_classes = Y_train.shape[1]
Theta = np.zeros((n_features, n_classes))

lr = 0.5
iters = 10000

Theta, cost_history = gradient_descent(X_train, Y_train, Theta, lr, iters)


plt.figure(figsize=(8,5))
plt.plot(cost_history, color='blue')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent Cost Convergence (Training Set)")
plt.grid(True)
plt.show()

def predict(X, Theta):
    logits = X @ Theta
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

y_train_pred = predict(X_train, Theta)
y_train_true = np.argmax(Y_train, axis=1)
train_accuracy = np.mean(y_train_pred == y_train_true)
print(f"Training accuracy: {train_accuracy*100:.2f}%")

y_test_pred = predict(X_test, Theta)
y_test_true = np.argmax(Y_test, axis=1)
test_accuracy = np.mean(y_test_pred == y_test_true)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

