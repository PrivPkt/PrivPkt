# -*- coding: utf-8 -*
#import socket module 
from socket import * 
import sys
import re
import ssl
import traceback

##########################################################
#global params
##########################################################
HOST = "127.0.0.1"
PORT = 8800
FILE = "index.html"
ssl_version = None
certfile = "./ssl/certificate.pem"
keyfile = "./ssl/key.pem"
ciphers = None
option_test_switch = 0 # to test, change to 1

version_dict = {
    "tlsv1.0" : ssl.PROTOCOL_TLSv1,
    "tlsv1.1" : ssl.PROTOCOL_TLSv1_1,
    "tlsv1.2" : ssl.PROTOCOL_TLSv1_2,
    "sslv23"  : ssl.PROTOCOL_SSLv23,
    "sslv3"   : ssl.PROTOCOL_SSLv3,
}


##########################################################
# Param Hander: get sslContext options through user input
##########################################################
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
    print "ver=", ssl_version, "ciphers=",ciphers, "certfile=", certfile, \
            "keyfile=", keyfile, "HOST=", HOST, "PORT=", PORT, "FILE=", FILE

##########################################################
# Init and configure SSLContext, then Wrap socket
# Params: socket sock
#         str ssl_version
#         str keyfile
#         str certificate
#         str ciphers
# Exception: SSLError
##########################################################
def ssl_wrap_socket(sock, ssl_version=None, keyfile=None, certfile=None, ciphers=None):

    #1. init a context with given version(if any)
    if ssl_version is not None and ssl_version in version_dict:
        #create a new SSL context with specified TLS version
        sslContext = ssl.SSLContext(version_dict[ssl_version])
        if option_test_switch == 1:
            print "ssl_version loaded!! =", ssl_version
    else:
        #if not specified, default
        sslContext = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
    if ciphers is not None:
        #if specified, set certain ciphersuite
        sslContext.set_ciphers(ciphers)
        if option_test_switch == 1:
            print "ciphers loaded!! =", ciphers
    
    #server-side must load certfile and keyfile, so no if-else
    sslContext.load_cert_chain(certfile, keyfile)
    print "ssl loaded!! certfile=", certfile, "keyfile=", keyfile
    
    try:
        return sslContext.wrap_socket(sock, server_side = True)
    except ssl.SSLError as e:
        print "wrap socket failed!"
        print traceback.format_exc()


#4. Prepare a sever socket 
serverSocket = socket(AF_INET, SOCK_STREAM) 
serverSocket.bind((HOST, eval(PORT)))
serverSocket.listen(10)


#######################################################
# Init socket and start connection (from hw1)
#######################################################
while True:
    #Establish the connection
    print 'Ready to serve...' 
    newSocket, addr = serverSocket.accept()
    connectionSocket = ssl_wrap_socket(newSocket, ssl_version, keyfile, certfile, ciphers)
    if not connectionSocket:
        continue
    
    try:
      message = connectionSocket.recv(1024)
      print "message=", message
      filename = message.split()[1] 
      print "filename=", filename
      f = open(filename[1:])  
      outputdata = f.read() 
      f.close()

      #Send one HTTP header line into socket
      #refrence website: https://goo.gl/UGTC9Q 
      response_headers = {
            'Content-Type': 'text/html; encoding=utf8',
            'Content-Length': len(outputdata),
            'Connection': 'close',
      }
      response_headers_raw = ''.join('%s: %s\n' % (k, v) for k, v in \
                                                response_headers.iteritems())
      connectionSocket.send('HTTP/1.1 200 OK')
      connectionSocket.send(response_headers_raw)
      connectionSocket.send('\n')
      
      #Send the content of the requested file to the client 
      connectionSocket.send(outputdata)
      
      #close socket after sending
      connectionSocket.shutdown(SHUT_RDWR)
      connectionSocket.close()

    except IOError:
        #Send response message for file not found 
        connectionSocket.send("404 Not Found")

        #Close client socket
        connectionSocket.shutdown(SHUT_RDWR)
        connectionSocket.close()
    
serverSocket.close()
sys.exit(0)
