#import py torch
#import efficientnet from pre trained model

import torch

#Import efficientnet
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)

#find the model summary
print(model)

#find the time taken by the model to run
print(mean_syn)

#find the standard deviation of the time taken by the model to run
print(std_syn)

#export the model to onnx
torch.onnx.export(model, dummy_input, "efficientnet-b0.onnx", verbose=True)