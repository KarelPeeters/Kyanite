from util import load_data, GoogleData

generic_data = load_data("../../data/loop/small_games.csv", shuffle=False)
data = GoogleData.from_generic(generic_data).to("cuda")

batch = data.pick_batch(slice(0, 10))
batch_rand = batch.random_symmetry()

print(batch.input[0, 0, :, :])
print(batch_rand.input[0, 0, :, :])
