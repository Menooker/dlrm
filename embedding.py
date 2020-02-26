# numpy
import numpy as np
# pytorch
import torch
import torch.nn as nn

np.random.seed(0)
n = 20000
m = 128
batch = 2048
lr = 0.001
fuse = 1

EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
W = np.random.uniform(
    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
).astype(np.float32)
EE.weight.data = torch.tensor(W, requires_grad=True)

EEB = nn.Embedding
# print(EE.weight.data)

if fuse:
  with torch.autograd.profiler.profile() as prof:
    for i in range(0, 100):
      emb_input = np.random.uniform(low=0, high=n, size=(batch)).astype(np.int64)
      emb_input = torch.LongTensor(emb_input)
      emb_offset = torch.LongTensor(np.linspace(0, batch, batch, dtype=np.int64))
      out = EE(emb_input, emb_offset).detach()
      out.requires_grad = True
      sum_out = torch.sum(out)
      sum_out.backward()
      EE.weight.data = nn.functional.embedding_bag_backward_sgd(
          EE.weight.data, lr, out.grad, emb_input, emb_offset, mode="sum")
      # print(x.grad)
else:
  optimizer = torch.optim.SGD(EE.parameters(), lr=lr)
  with torch.autograd.profiler.profile() as prof:
    for i in range(0, 100):
      emb_input = np.random.uniform(low=0, high=n, size=(batch)).astype(np.int64)
      emb_input = torch.LongTensor(emb_input)
      emb_offset = torch.LongTensor(np.linspace(0, batch, batch, dtype=np.int64))
      optimizer.zero_grad()
      out = EE(emb_input, emb_offset)
      sum_out = torch.sum(out)
      sum_out.backward()
      optimizer.step()
# print(EE.weight.data)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
