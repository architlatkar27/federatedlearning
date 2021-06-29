import numpy
from collections import Counter
import pandas as pd

import socket
import pickle
from threading import *
from time import sleep, time
import struct
import random

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack('4sl', socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
print("client connected")

print("multicast client connected")

host = 'localhost'
port = 5008


client_model = random.randint(1, 30)

while True:
    '''
    step 1 - check the accuracy (here we will generate random numbers)
    step 2 - send model to server if accuracy is low
    step 3 - recv new model from server
    step 4 - sleep for 5 secs XD
    '''
    #step 1
    x = random.randint(0, 100)
    print("x: {}".format(x))
    if x > client_model:
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock1.connect((host, port))
        print("client connected to server")
        sock1.send(str(client_model).encode("ascii"))
        print("client sent model to server")
        sock1.close()

    y = sock.recv(1024)
    print("got y: {} of type {}".format(y, type(y)))
    y = int(y)
    client_model = y
    print("client model updated from server")
    print("model: {}".format(client_model))
    client_model += random.randint(-5, 5)
    sleep(5)

