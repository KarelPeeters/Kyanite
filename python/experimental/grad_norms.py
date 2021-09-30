from matplotlib import pyplot
from torch import nn

from lib.train import TrainSettings


def plot_grad_norms(settings: TrainSettings, network: nn.Module, batch):
    act_norms = []
    grad_norms = []

    def forward_hook(module, _, output):
        if not isinstance(output, tuple):
            output = (output,)

        for output in output:
            act_norms.append((type(module), float((output ** 2).mean())))
        print("forward", module)

    def backward_hook(module, _, grad_output):
        if not isinstance(grad_output, tuple):
            grad_output = (grad_output,)

        for grad_output in grad_output:
            grad_norms.append((type(module), float((grad_output ** 2).mean())))
        print("backward", module)

    handles = []

    for _, module in network.named_modules():
        handles.append(module.register_forward_hook(forward_hook))
        handles.append(module.register_full_backward_hook(backward_hook))

    def dummy_log(*args):
        pass

    network.train()
    loss = settings.evaluate_loss(network, "", dummy_log, batch)
    loss.backward()

    grad_norms = list(reversed(grad_norms))

    for handle in handles:
        handle.remove()

    all_types = {ty for ty, _ in act_norms}

    def plot_per_type(data):
        for ty in all_types:
            indices = []
            values = []
            for i, (actual_ty, v) in enumerate(act_norms):
                if ty == actual_ty:
                    indices.append(i)
                    values.append(v)
            pyplot.plot(indices, values, label=ty.__name__)

    pyplot.figure()
    plot_per_type(act_norms)
    pyplot.legend()
    pyplot.title("Activation norms")
    pyplot.show()

    pyplot.figure()
    plot_per_type(grad_norms)
    pyplot.legend()
    pyplot.title("Gradient norms")
    pyplot.ylim(0, 100)
    pyplot.show()
