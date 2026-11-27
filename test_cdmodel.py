import torch

from cdmodelv3 import CDModel, PriorGuidedChangeFusion


def print_shape(name, tensor):
    print(f"{name}: {tuple(tensor.shape)}")


def main():
    torch.manual_seed(0)

    batch_size = 1
    in_channel = 1
    out_channel = 1
    height = 4
    width = 4

    xA = torch.randn(batch_size, in_channel, height, width)
    xB = torch.randn(batch_size, in_channel, height, width)
    x_context = torch.randn(batch_size, out_channel, height // 2, width // 2)

    model = CDModel(in_channel=in_channel, out_channel=out_channel)
    fusion = PriorGuidedChangeFusion(in_channel=in_channel, out_channel=out_channel)
    model.eval()
    fusion.eval()

    with torch.no_grad():
        y = model(xA, xB, x_context)
        y_without_context = model(xA, xB)
        prior_feat = torch.randn_like(y)
        fused, gate = fusion(prior_feat, y, return_gate=True)

    print("CDModel input/output shapes")
    print_shape("xA", xA)
    print_shape("xB", xB)
    print_shape("x_context", x_context)
    print_shape("output", y)
    print_shape("output_without_context", y_without_context)
    print_shape("prior_feat", prior_feat)
    print_shape("fused", fused)
    print_shape("gate", gate)


if __name__ == "__main__":
    main()
