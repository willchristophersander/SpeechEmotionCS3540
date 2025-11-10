import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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


Y = data[['Emotion_num']]

Y = Y.to_numpy().reshape(-1,1)

print(Y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

m, n = Y.shape
K = len(np.unique(Y))

Y_onehot = np.zeros((m, K))
for i in range(m):
    Y_onehot[i, Y[i]] = 1

print(Y_onehot)

x_train, x_test, y_train, y_test = train_test_split(X, Y_onehot, test_size = 0.25, random_state = 0)

Y_onehot = Y_onehot.T

print(Y_onehot.shape)


print(X.shape, Y.shape)

Y = Y.T

print(Y.shape)
print(Y)

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print("Reduced features:", X_reduced.shape[1])

X = X.T

X_reduced = X_reduced.T

print(X_reduced.shape)

print(Y_onehot.shape)



def layerSizes(X):
    input_layer_size = X.shape[0]
    hidden_layer1_size = 6
    hidden_layer2_size = 6
    output_layer_size = 6

    return input_layer_size, hidden_layer1_size, hidden_layer2_size, output_layer_size
def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    Weight1 = np.random.rand(hidden1_size,(input_size+1))
    Weight2 = np.random.rand(hidden2_size,(hidden1_size+1))
    Weight3 = np.random.rand(output_size, (hidden2_size + 1))
    parameters = {'Weight1' : Weight1, 'Weight2' : Weight2, 'Weight3' : Weight3}

    return parameters
input , h1, h2, output = layerSizes(X)

parameters = (initialize_parameters(input , h1, h2, output))

print(parameters['Weight1'].shape)
print(parameters['Weight2'].shape)
print(parameters['Weight3'].shape)




def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)
def forward_prop(X, parameters):
    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    Weight3 = parameters['Weight3']

    one = np.ones(X.shape[1])
    X = np.vstack((one, X))

    z2 = np.dot(Weight1, X)

    a2 = relu(z2)

    one = np.ones(a2.shape[1])
    a2  = np.vstack((one,a2))

    z3 = np.dot(Weight2, a2)

    a3 = relu(z3)

    one = np.ones(a3.shape[1])
    a3  = np.vstack((one,a3))

    z4 = np.dot(Weight3, a3)

    a4 = softmax(z4)

    cache = {"z2": z2,
             "a2": a2,
             "z3": z3,
             "a3": a3,
             "z4": z4,
             "a4": a4}
    return np.array(a4), cache


    

input , h1, h2, output = layerSizes(X)

parameters = (initialize_parameters(input , h1, h2, output))

print(parameters['Weight1'].shape)
print(parameters['Weight2'].shape)
print(parameters['Weight3'].shape)

a4, cache = forward_prop(X, parameters)

print(a4, cache)
def compute_cost(a4, Y):

    m = Y.shape[1]


    cost = -(1 / m) * np.sum(Y * np.log(a4 + 0.000000008))

    return np.squeeze(cost)
def backward_prop(parameters, cache, X, Y):

    m = X.shape[1]

    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    Weight3 = parameters['Weight3']

    one = np.ones(X.shape[1])
    X = np.vstack((one, X))

    z2 = cache['z2']
    a2 = cache['a2']
    z3 = cache['z3']
    a3 = cache['a3']
    a4 = cache['a4']

    d4 = a4 - Y
    dW3 = d4 @ a3.T

    d3 = Weight3.T @ d4
    d3 = d3[1:]
    d3 = d3 * relu_derivative(z3)

    dW2 = d3 @ a2.T

    d2 = Weight2.T @ d3
    d2 = d2[1:]
    d2 = d2 * relu_derivative(z2)

    dW1 = d2 @ X.T

    grads = {"dWeight1": 1/m*dW1,
             "dWeight2": 1/m*dW2,
             "dWeight3": 1/m*dW3}
    return grads
input , h1, h2, output = layerSizes(X)

parameters = (initialize_parameters(input , h1, h2, output))

print(parameters['Weight1'].shape)
print(parameters['Weight2'].shape)
print(parameters['Weight3'].shape)

a4, cache = forward_prop(X, parameters)


grads = backward_prop(parameters, cache, X, Y_onehot)

print ("dW1 = " + str(grads["dW1"]))
print ("dW2 = " + str(grads["dW2"]))
def update_parameters(parameters, grads, learning_rate):

    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    Weight3 = parameters['Weight3']

   
    dW1 = grads['dWeight1'] 
    dW2 = grads['dWeight2']
    dW3 = grads['dWeight3'] 
    
    Weight1 = Weight1 - (learning_rate * dW1)
    Weight2 = Weight2 - (learning_rate * dW2) 
    Weight3 = Weight3 - (learning_rate * dW3)

    parameters = {"Weight1": Weight1,
                  "Weight2": Weight2,
                  "Weight3": Weight3}

    return parameters
def initialize_adam(parameters):
    v = {}
    s = {}

    for key in parameters.keys():
        v["d" + key] = np.zeros_like(parameters[key])
        s["d" + key] = np.zeros_like(parameters[key])

    return v, s
def update_parameters_adam(parameters, grads, v, s, t, 
                           learning_rate=0.001, 
                           beta1=0.9, beta2=0.999, epsilon=1e-8):
    for key in parameters.keys():
        # first moment
        v["d" + key] = beta1 * v["d" + key] + (1 - beta1) * grads["d" + key]
        # second moment
        s["d" + key] = beta2 * s["d" + key] + (1 - beta2) * np.square(grads["d" + key])
        # Bias correction
        v_corrected = v["d" + key] / (1 - beta1 ** t)
        s_corrected = s["d" + key] / (1 - beta2 ** t)
        # Parameter update
        parameters[key] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return parameters, v, s
def model(X, Y, n_h1, n_h2, num_epochs=1000, learning_rate=0.1, print_cost=True, print_graph=True):
 
    cost_data = []

    itters = []
    
    n_x = X.shape[0]

    if Y.ndim == 2:
        n_y = Y.shape[0]  
    else:
        n_y = len(np.unique(Y))


  
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    v, s = initialize_adam(parameters)
    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    Weight3 = parameters['Weight3']

    for i in range(num_epochs):
        
        t = i + 1

        a4, cache = forward_prop(X, parameters)

        cost = compute_cost(a4, Y)

        grads = backward_prop(parameters, cache, X, Y)

        parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate)
        # parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_graph:
            cost_data.append(cost)
            itters.append(i)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            
            
    if print_graph:
        plt.xlabel = "itters"
        plt.ylabel = "cost"   
        plt.plot(itters, cost_data)
            

    return parameters
def predict(parameters, X):
  
    a4, cache = forward_prop(X, parameters)

    predictions = np.argmax(a4, axis=0)

    return predictions
parameters = model(x_train.T, y_train.T, 6, 6, 10000, learning_rate = .1, print_cost = False)

predictions = predict(parameters, x_test.T)


y_true = np.argmax(y_test, axis=1)
print(accuracy_score(y_true, predictions))

X_first100 = X[:, :100]
Y_first100 = Y[:, :100]
predictions = predict(parameters, X_first100)
print(accuracy_score(Y_first100.T, predictions))
# first 100 predictions using trained model.
print(predictions)
# first 100 true y for visualization of results
print(Y.T[:100])
