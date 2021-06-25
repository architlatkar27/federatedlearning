import socket
import time
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
