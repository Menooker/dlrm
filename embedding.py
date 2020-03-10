# numpy
import numpy as np
# pytorch
import torch
import torch.nn as nn
import sys
import time
import argparse

### parse arguments ###
parser = argparse.ArgumentParser(
    description="Train Deep Learning Recommendation Model (DLRM)"
)
# model related parameters
parser.add_argument("--m", type=int, default=128)
parser.add_argument("--n", type=int, default=4000000)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--fuse", type=int, default=0)
parser.add_argument("--mode", type=int, default=0)
args = parser.parse_args()



n = args.n
m = args.m
batch = args.batch
lr = 0.001
fuse = args.fuse
mode = args.mode

# print(EE.weight.data)
t0=time.time()
timesum=0
def orig_sgd():
  np.random.seed(0)
  EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
  W = np.random.uniform(
      low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
  ).astype(np.float32)
  EE.weight.data = torch.tensor(W, requires_grad=True)

  EEB = nn.Embedding
  optimizer = torch.optim.SGD(EE.parameters(), lr=lr)
  for i in range(0, 20):
      emb_input = np.random.uniform(low=0, high=n, size=(batch)).astype(np.int64)
      emb_input = torch.LongTensor(emb_input)
      emb_offset = torch.LongTensor(np.linspace(0, batch, batch, dtype=np.int64))
      optimizer.zero_grad()
      out = EE(emb_input, emb_offset)
      sum_out = torch.sum(out)
      sum_out.backward()
      optimizer.step()
  return EE.weight.data

def fused_sgd():
  np.random.seed(0)
  EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
  W = np.random.uniform(
      low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
  ).astype(np.float32)
  EE.weight.data = torch.tensor(W, requires_grad=True)

  EEB = nn.Embedding
  for i in range(0, 20):
      emb_input = np.random.uniform(low=0, high=n, size=(batch)).astype(np.int64)
      emb_input = torch.LongTensor(emb_input)
      emb_offset = torch.LongTensor(np.linspace(0, batch, batch, dtype=np.int64))
      out = EE(emb_input, emb_offset).detach()
      out.requires_grad = True
      sum_out = torch.sum(out)
      sum_out.backward()
      sgdfunc = nn.functional.embedding_bag_backward_sgd3 if fuse==3 else nn.functional.embedding_bag_backward_sgd
      #time.sleep(20)
      global timesum
      t00=time.time()
      #print("Sleep done2")
      #for i in range(0, 10000):
      EE.weight.data=sgdfunc(
            EE.weight.data, lr, out.grad, emb_input, emb_offset, mode="sum")
      timesum+= (time.time()-t00)
      # print(x.grad)
  return EE.weight.data

with torch.autograd.profiler.profile() as prof:
  if fuse:
    newW = fused_sgd()
  else:
    newW = orig_sgd()
t1=time.time()
if mode==0:
  print(newW)
  print(newW.sum())
  print(prof.key_averages().table(sort_by="self_cpu_time_total"))
  print("TIME",t1-t0)
  print(timesum)
elif mode==1:
  print("Compute done")
  torch.save(newW, "saved.{}.{}.{}.pt".format(m,n,batch))
elif mode==2:
  print("Compute done")
  gt = torch.load("saved.{}.{}.{}.pt".format(m,n,batch))
  print("Load done")
  print(torch.allclose(gt, newW))
elif mode==3:
  print("Compute done. Computing ground truth")
  result = newW
  cmpresult = orig_sgd()
  print("Load done")
  print(torch.allclose(cmpresult, result))