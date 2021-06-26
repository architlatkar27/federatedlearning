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
print("server connected")

sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock1.bind("localhost", 5008)
sock1.listen(5)

#helper functions
arr = []
def aggregator():
    return sum(arr)/len(arr)
def recv_threaded_client(connection, lock):
    data = connection.recv(100000)
    lock.acquire()
    arr.append(int(str(data)))
    lock.release()
    connection.close()

con_threads = []
lock = Lock()

while True:
    '''
    step1 - send getmodel command

    step2 - receive model parameters from the clients

    step3 - create an aggregate model

    step4 - send recvmodel command

    step5 - send aggregate model params
    '''
    #step 1
    sock.sendto(b'getmodel', (MCAST_GRP, MCAST_PORT))

    #step 2
    start_time = time()
    limit = 10
    while True:
        if time()-start_time > limit:
            break
        Client, address = sock1.accept()
        con_threads.append(Thread(target=recv_threaded_client, args=(Client, lock)))
        con_threads[-1].start()
    
    for t in con_threads:
        t.join()
    
    #hopefully, we have acquired models so time for step 3
    new_model = aggregator()

    #step 4
    sock.sendto(b'recvmodel', (MCAST_GRP, MCAST_PORT))
    #step 5
    sock.sendto(str(new_model).encode(), (MCAST_GRP, MCAST_PORT))
    
    sleep(5)



