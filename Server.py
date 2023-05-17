# import socket
# import ssl
# import struct
#
# # Define the server's IP address and port number
# IP = '127.0.0.1'
# PORT = 1234
#
# # Create a socket object
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# # Bind the socket to a specific IP address and port number
# server_socket.bind((IP, PORT))
#
# # Listen for incoming connections
# server_socket.listen()
#
# # Load SSL certificates
# context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
# context.load_cert_chain(certfile='server.crt', keyfile='server.key')
#
# # Accept incoming connections and receive data
# while True:
#     client_socket, client_address = server_socket.accept()
#     ssl_client_socket = context.wrap_socket(client_socket, server_side=True)
#
#     # Receive data from the client
#     data = ssl_client_socket.recv(1024)
#
#     # Unpack the received data
#     stddev = struct.unpack('f', data)[0]
#
#     # Print the received standard deviation
#     print(f"Received standard deviation from client {client_address}: {stddev}")
#
#     # Close the SSL client socket
#     ssl_client_socket.close()
