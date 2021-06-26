import socket
import pickle
import threading
import numpy
from collections import Counter
import pandas as pd

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


class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        nn = NeuralNet()  # create the NN model
        Gparams = nn.params.copy()
        msg = pickle.dumps(Gparams)
        connection.sendall(msg)
        print("Server sent intial global message to the client.")

    def recv(self):
        received_data = b""
        while connection:
            try:
                data = connection.recv(self.buffer_size)
                received_data += data

                if data == b'':  # Nothing received from the client.
                    received_data = b""
                    return None, 0  # 0 means the connection is no longer active and it should be closed.
                elif str(data)[-2] == '.':
                    print(
                        "All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info,
                                                                                          data_len=len(received_data)))
                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def run(self):
        while True:
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(
                    "Connection Closed with {client_info} either due to inactivity for seconds or due to an error.".format(
                        client_info=self.client_info), end="\n\n")
                break

            print("Mean params: client.")
            keys = set(list(Gparams.keys()) + list(received_data.keys()))
            D = {key: (Gparams.get(key, 0) + received_data.get(key, 0) ) / float((key in Gparams) + (key in received_data) ) for
                 key in keys}
            print(D)

            msg = pickle.dumps(D)
            connection.sendall(msg)
            print("Server sent a message to the client.")

soc = socket.socket()
print("Socket is created.")

soc.bind(("localhost", 10000))
print("Socket is bound to an address & port number.")

soc.listen(10)
print("Listening for incoming connection ...")

# while True:
#     try:
#         connection, client_info = soc.accept()
#         print("New Connection from {client_info}.".format(client_info=client_info))

#         socket_thread = SocketThread(connection=connection,
#                                      client_info=client_info,
#                                      buffer_size=1024)
#         socket_thread.start()
#     except:
#         soc.close()
#         print(" Socket Closed Because no Connections Received.\n")
#         break

import time


def multicast(text):
    MCAST_GRP = '224.1.1.1'
    MCAST_PORT = 5007

    cmd = b'recvmodel'
    data = b'some data sent by server'

    MULTICAST_TTL = 255

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
    print("server connected")
    while True:
        sock.sendto(cmd, (MCAST_GRP, MCAST_PORT))
        sock.sendto(data, (MCAST_GRP, MCAST_PORT))
        print("message sent")
        # rec = sock.recv(1000)
        # print(rec)

        time.sleep(5)