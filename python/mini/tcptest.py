import socket

while True:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.connect(("::1", 63105))
        print(s)

        break
    except ConnectionRefusedError as e:
        print(e)

s.send(b'{"NewSettings":{"games_per_file":1000,"max_game_length":400,"exploration_weight":2.0,"random_symmetries":true,"keep_tree":false,"temperature":1.0,"zero_temp_move_count":20,"dirichlet_alpha":0.25,"dirichlet_eps":0.2,"full_search_prob":1.0,"full_iterations":500,"part_iterations":500,"cache_size":100000}}\n')
s.send(b'{"NewNetwork":"C:/Documents/Programming/STTT/AlphaZero/data/derp/basic_res_model/params.npz"}\n')
# s.send(b'"Stop"\n')

# keep socket open
while True:
    pass