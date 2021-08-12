import time

from selfplay_client import SelfplayClient, SelfplaySettings

while True:
    client = SelfplayClient()
    settings = SelfplaySettings(
        max_game_length=10,
        exploration_weight=2.0,
        random_symmetries=True,
        keep_tree=False,
        temperature=1.0,
        zero_temp_move_count=20,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        full_iterations=2,
        part_iterations=100,
        cache_size=0
    )
    client.send_new_settings(settings)
    client.send_new_network("C:/Documents/Programming/STTT/AlphaZero/data/derp/basic_res_model/model.onnx")

    # client.send_stop()
    while True:
        print(client.wait_for_file())

    time.sleep(0.1)
