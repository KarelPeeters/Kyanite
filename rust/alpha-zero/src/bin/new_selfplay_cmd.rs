use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::games::ataxx_output::AtaxxBinaryOutput;
use alpha_zero::new_selfplay::server::selfplay_server_main;
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let load_network = |path: String, batch_size: usize, device: Device| -> AtaxxCNNNetwork {
        AtaxxCNNNetwork::load(path, batch_size, device)
    };

    let output = |path: &str| AtaxxBinaryOutput::new(path);

    selfplay_server_main("ataxx", AtaxxBoard::new_without_gaps, output, load_network);
}
