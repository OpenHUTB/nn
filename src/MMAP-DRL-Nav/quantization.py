import torch

def quantize_model(model): #对模型进行动态量化
    model.eval()
    with torch.no_grad():
        quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model
