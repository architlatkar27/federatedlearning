import socket
import time
import struct
'''
server sends getmodel ON Multicast group

receiver sends model file on tcp p2p

---Server side model is aggregated---

server sends recvmodel ON multicast grp

receiver prepares to accept model

sender sends model on Multicast grp

receiver recvs model and updates itself

'''

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack('4sl', socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
print("client connected")

while True:
    cmd = sock.recv(1000)
    if cmd == b'getmodel':
        #write code to send data on tcp connection
        print("sending model to server")
        sock.sendto(b'sending model', (MCAST_GRP, MCAST_PORT))
    elif cmd == b'recvmodel':
        #write code to receive and update model on tcp connection
        print("updating model from server")
        model = sock.recv(1000)
        print(model)
    else:
        print(cmd)

    time.sleep(5)