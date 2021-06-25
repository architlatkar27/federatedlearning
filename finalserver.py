import numpy
from collections import Counter
import pandas as pd

import socket
import pickle
import threading
from time import sleep

Gparams = {}

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
        numpy.random.seed(2)  # Seed the random number generator
        self.params["W1"] = numpy.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = numpy.random.randn(self.layers[1], )
        self.params['W2'] = numpy.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = numpy.random.randn(self.layers[2], )
        print("Created")
        print(self.params)


MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

cmd = b'recvmodel'
data = b'some data sent by server'

MULTICAST_TTL = 255

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
print("server connected")
while True:
    '''
    step1 - send getmodel command

    step2 - receive model parameters from the clients

    step3 - create an aggregate model

    step4 - send recvmodel command

    step5 - send aggregate model params
    '''
    pass

    sleep(5)