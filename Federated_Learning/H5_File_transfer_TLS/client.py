# -*- coding: utf-8 -*
from socket import * 
import sys
import re
import ssl
import pprint
import traceback

######################################
# Global params
######################################
HOST = "127.0.0.1"
PORT = 8800
FILE = "index.html"
ssl_version = None
certfile = None
keyfile = "./ssl/key.pem"
ciphers = None
hostname = "localhost"
option_test_switch = 0 # to test, change to 1

version_dict = {
    "tlsv1.0" : ssl.PROTOCOL_TLSv1,
    "tlsv1.1" : ssl.PROTOCOL_TLSv1_1,
    "tlsv1.2" : ssl.PROTOCOL_TLSv1_2,
    "sslv23"  : ssl.PROTOCOL_SSLv23,
    "sslv3"   : ssl.PROTOCOL_SSLv3,
}

###########################################################
# Param handler: get sslContext options through user input
###########################################################
for i in range(1, len(sys.argv)):
    arg = sys.argv[i]
    if re.match("[-]{,2}(tlsv|sslv)[0-9.]{,3}", arg, re.I):
        ssl_version = re.sub("-", "", arg)
    if re.match("[-]{,2}ciphers", arg, re.I):
        ciphers = sys.argv[i + 1]
    if re.match("[-]{,2}cacert", arg, re.I):
        certfile = sys.argv[i + 1]
    if re.match("^[0-9]{,3}\.[0-9]{,3}\.[0-9]{,3}\.[0-9]{,3}|localhost$", arg, re.I):
        HOST = arg
    if re.match("^[0-9]{,5}$", arg):
        PORT = arg
    if re.match("^[0-9a-zA-Z_/]+\.[0-9a-zA-Z-_/]+$", arg, re.I):
        FILE = arg

if option_test_switch == 1:
    print "ver=", ssl_version, "ciphers=",ciphers, "certfile=", certfile, "keyfile=", \
            keyfile, "HOST=", HOST, "PORT=", PORT, "FILE=", FILE

################################################################################
# Init and configure SSLContext, then Wrap socket in context
# Params: socket sock
#         str ssl_version
#         str keyfile
#         str certificate
#         str ciphers
# Note: For client-side sockets, the context construction is lazy 
#       if the underlying socket isn't connected yet, the context 
#       construction will be performed after connect() is called on the socket.
# Exception: SSLError
###############################################################################
def ssl_wrap_socket(sock, ssl_version=None, keyfile=None, certfile=None, ciphers=None):
    try:
        #2.init a sslContext with given version (if any)
        if ssl_version is not None and ssl_version in version_dict:
            #create a new SSL context with specified TLS version
            sslContext = ssl.SSLContext(version_dict[ssl_version])
            if option_test_switch == 1:
                print "ssl_version loaded!! =", ssl_version
        else:
            #default
            sslContext = ssl.create_default_context()
        
        if ciphers is not None:
            #if specified, set certain ciphersuite
            sslContext.set_ciphers(ciphers)
            if option_test_switch == 1:
                print "ciphers loaded!! =", ciphers
        
        #3. set root certificate path
        if certfile is not None and keyfile is not None:
            #if specified, load speficied certificate file and private key file
            sslContext.verify_mode = ssl.CERT_REQUIRED
            sslContext.check_hostname = True
            sslContext.load_verify_locations(certfile, keyfile)
            if option_test_switch == 1:
                print "ssl loaded!! certfile=", certfile, "keyfile=", keyfile 
            return sslContext.wrap_socket(sock, server_hostname = hostname)
        else:
            #default certs
            sslContext.check_hostname = False
            sslContext.verify_mode = ssl.CERT_NONE
            sslContext.load_default_certs()
            return sslContext.wrap_socket(sock)
        
    except ssl.SSLError:
        print "wrap socket failed!"
        print traceback.format_exc()
        sock.close()
        sys.exit(-1)


######################################
# Connection related (from hw1)
######################################

#4.Prepare a client socket
clientSocket = socket(AF_INET, SOCK_STREAM)

#5.Wrapping the TCP socket with the SSL/TLS context
sslSocket = ssl_wrap_socket(clientSocket, ssl_version, keyfile, certfile, ciphers)

#6.connect to server
sslSocket.connect((HOST, eval(PORT)))

try:
    #prepare HTTP header
    message = "GET /" + FILE + " HTTP/1.1\r\n\r\n"
    
    #Send the whole string
    sslSocket.sendall(message)

    #receive data
    while sslSocket.recv(1024):
        reply = sslSocket.recv(1024)
        if(certfile is None):
            print reply
        else:
            #part 3-print certificate
            pprint.pprint(sslSocket.getpeercert())

except socket.error:
    #Send failed
    print 'ERROR: Send failed'
    sslSocket.shutdown(SHUT_RDWR)
    sslSocket.close()
    sys.exit(-1)

finally:
    #7.close the socket
    sslSocket.close()
    sys.exit(0)
