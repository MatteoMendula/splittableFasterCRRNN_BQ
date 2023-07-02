import torch
from dumbdumbmodel import stupidassmodel
torch.backends.cudnn.enabled = False


loaded_model = stupidassmodel()
loaded_model.load_state_dict(torch.load('stupidmodel.pt'))
loaded_model.cuda()
loaded_model.eval()
print("loaded! :D")

input = torch.randn(1, 3, 32, 32)
input = input.cuda()

torch.onnx.export(
    loaded_model,
    input,
    "stupidmodel13.onnx",
    verbose=False,
    opset_version=13,
    do_constant_folding = False)