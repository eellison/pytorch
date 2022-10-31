class model(torch.nn.Module):
    def forward(self, a_1: f32[100, 100], b_1: f32[100, 100]):
        # No stacktrace found for following nodes
        add: f32[100, 100] = torch.ops.aten.add.Tensor(a_1, b_1);  a_1 = b_1 = None
        return (add,)
        