import numpy
from collections import Counter
import pandas as pd

import socket
import pickle
from threading import *
from time import sleep, time


MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
MULTICAST_TTL = 255

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
print("multicast server connected")

sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock1.bind(("localhost", 5008))
sock1.listen(5)

#helper functions
our_model = 10
def aggregator(new_model):
    global our_model
    avg = (our_model+new_model)//2
    our_model = avg
    # return avg

def recv_threaded_client(connection, lock):
    #step 2
    print("request for aggregation received")
    data = connection.recv(100000)
    print("got model: {}".format(data))
    lock.acquire()
    aggregator(int(data)) # here, we must parse the actual model and then pass it on. Taking a number for testing purpose
    sock.sendto(str(our_model).encode("ascii"), (MCAST_GRP, MCAST_PORT))
    print("new model is: {}".format(our_model))
    lock.release()
    connection.close()


con_threads = []
lock = Lock()

while True:
    '''
    step1 receive client's model
    step2 generate aggregate model
    step3 broadcast aggregate model to all clients

    '''
    #step 1
    Client, address = sock1.accept()
    t1 = Thread(target=recv_threaded_client, args=(Client, lock))
    t1.start()
    print("aggregation done")
    #step 3
    #sock.sendto(str(our_model).encode("ascii"), (MCAST_GRP, MCAST_PORT))
    print("new model multicasted to all clients")




