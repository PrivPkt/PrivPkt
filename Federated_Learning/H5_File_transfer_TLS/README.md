# SSL/TLS Secure Socket Programming (Python)

This is a simple implementation of SSL/TLS web socket (HTTPS) on both client and server end. Keys, certificates and ciphers are only for test use.

## Notes:
* support user selecting TSL versions, ciphersuites, CA certificate file path as parameters (both client and server end)
* support retrieving TLS/SSL certificate info from web server
* testing certificates are self-assigned with openssl
* use LAN address for testing
* python version 2.7.13, openssl version 1.0.2 

## To init a SSL connection:
### Server: 
	$ python server.py host port file

ex run:

**Exemple 1. with no specified configuration:**
    
	python server.py 127.0.0.1 8801 index.html
			
**Exemple 2. with cert path:**

	python server.py --cacert ./ssl/certificate.pem 127.0.0.1 8801 index.html
			
**Exemple 3. with version:**  

	python server.py --sslv23 --cacert ./ssl/certificate.pem 127.0.0.1 8801 index.html
			
**Exemple 4. with cipher:**
     
	python server.py --sslv23 --cacert ./ssl/certificate.pem --cipher ECDHE-RSA-AES256-GCM-SHA384 127.0.0.1 8801 index.html



### Client:  
***Note: For the purpose of testing only the connection, on the client-side should not specify certificate file path (but DOES NOT mean this parameter won’t work, its effectiveness can be tested in the next part), otherwise will change to the result of certificate-printing (next part).***
 
	$ python client.py <ssl/tsl version> <ciphers> host port file

ex run:

**Exemple 1. no specification:**
          
	python client.py 127.0.0.1 8801 index.html

**Exemple 2. with version:** 
   
	python client.py --tlsv1.1 127.0.0.1 8801 index.html
			
**Exemple 3. with cipher:** 
      
	python client.py --ciphers ECDHE-RSA-AES256-GCM-SHA384 127.0.0.1 8801 index.html


## Part2 (specify cert file and retrieve server cert info):	
### Server: (no change)
### Client:
***NOTE: If client specifies certificate, then switch to this part, client-side print will be ONLY the certificate got from client***
	
	$ python client.py --cacert path <ssl/tsl version> <ciphers> host port file

ex run: 

**Exemple 1. with only cert:**

	python client.py --cacert ./ssl/certificate.pem 127.0.0.1 8801
	
**Exemple 2. with version:**

	python client.py --tlsv1.2 --cacert ./ssl/certificate.pem 127.0.0.1 8801

**Exemple 3. with ciphters:**

	python client.py --tlsv1.2 --ciphers ECDHE-RSA-AES256-GCM-SHA384 --cacert ./ssl/certificate.pem 127.0.0.1 8801  

  
### Other Note: 
***1.	To test that user input option params are correctly accepted and used, change the global variable “option_test_switch” value to 1 (can do on both client and server).***

***2.	All ssl files are stored in ./ssl/ directory***
