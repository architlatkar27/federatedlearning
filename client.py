  
import socket
import pickle

class NeuralNet():
    '''
    A two layer neural network
    '''

    def __init__(self, layers=[4, 4, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None


    def relu(self, Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def eta(self, x):
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1 / (1 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat)  ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''

        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = yhat * (yhat_inv)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        # update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()



soc = socket.socket()
print("Socket is created.")

soc.connect(("localhost", 10000))
print("Connected to the server.")
nn = NeuralNet()

received_data = b''
while str(received_data)[-2] != '.':
    data = soc.recv(1024)
    received_data += data
received_data = pickle.loads(received_data)
#print("Received Initial data from the server: {received_data}".format(received_data=received_data))
nn.params = received_data.copy()
print("Received Initial nn param data from the server:")
print(nn.params)

msg = pickle.dumps(nn.params)
soc.sendall(msg)
print("Client sent a nn params message to the server.")

received_data = b''
while str(received_data)[-2] != '.':
    data = soc.recv(64)
    received_data += data
received_data = pickle.loads(received_data)
#print("Received global params from the server: {received_data}".format(received_data=received_data))
nn.params = received_data.copy()
print("Received first global nn params data from the server:")
print(nn.params)

import csv
import pandas as pd

# add header names
headers =  ['Battery_Level','Temperature','Voltage', 'Current','Status']
heart_df = pd.read_csv('Charging.csv')
heart_df.head()

import numpy as np
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#convert imput to numpy arrays
X = heart_df.drop(columns=['Voltage'])
X = np.array(X).astype("float")

#replace target class with 0 and 1
#1 means "have heart disease" and 0 means "do not have heart disease"
#heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
#heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

y_label = heart_df['Voltage'].values.reshape(X.shape[0], 1)
y_label = np.array(y_label).astype("float")

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

nn.fit(Xtrain, ytrain) #train the model
print("A Input weight:", nn.params['W1'])
print("A Output weight:", nn.params['W2'])
print("A Input b1 weight:", nn.params['b1'])
print("A Input b2 weight:", nn.params['b2'])
msg = pickle.dumps(nn.params)
soc.sendall(msg)
print("Client sent trained local nn params message after training model to the server.")
print(nn.params['b2'])

received_data = b''
while str(received_data)[-2] != '.':
    data = soc.recv(1024)
    received_data += data
received_data = pickle.loads(received_data)
#print("Received global params from the server  after training model : {received_data}".format(received_data=received_data))
nn.params = received_data.copy()

print("Received global modified  nn params data from the server after training model :")
print(nn.params['b2'])

train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

print("BeforevTrain accuracy is {}".format(nn.acc(ytrain, train_pred)))
print("Before Test accuracy is {}".format(nn.acc(ytest, test_pred)))

soc.close()
print("Socket is closed.")