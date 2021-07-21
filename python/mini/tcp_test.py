import json
import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 8668))

print(s)

command = "Stop"
s.send(bytes(json.dumps(command) + "\n", "utf-8"))

while True:
    pass