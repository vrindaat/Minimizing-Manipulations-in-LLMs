import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel
from labels import *



def get_scores(prompt):
  class customModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.model=AutoModel.from_pretrained("microsoft/deberta-v3-base",cache_dir="/tmp/huggingface")
      self.model1=AutoModel.from_pretrained("roberta-base",cache_dir="/tmp/huggingface")
      self.linearLayer=nn.Linear(768,128)
      self.activation1=nn.ReLU()
      self.finalLayer=nn.Linear(128,146)

    def forward(self,inputId,attention,inputId_="",attention_=""):
      output1=self.model(inputId,attention)
      output2=self.model1(inputId_,attention_)
      output1=output1[0][:,0,:]
      output2=output2[0][:,0,:]
      output1=torch.add(output1,output2)
      output1=output1/2
      output1=self.linearLayer(output1)
      output1=self.activation1(output1)
      final=self.finalLayer(output1)
      return final

  model_ = customModel()
  model_.load_state_dict(torch.load("model-7.pt",map_location=torch.device('cpu')))

  tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base",force_download=True,cache_dir="/tmp/huggingface")
  tokenizer_=AutoTokenizer.from_pretrained("roberta-base",force_download=True,cache_dir="/tmp/huggingface")
  item=f"{prompt}"
  model_.to("cpu")
  input_ids,attention_mask=tokenizer([item],return_tensors="pt")["input_ids"].to("cpu"),tokenizer([item],return_tensors="pt")["attention_mask"].to("cpu")
  input_ids_,attention_mask_=tokenizer_([item],return_tensors="pt")["input_ids"].to("cpu"),tokenizer_([item],return_tensors="pt")["attention_mask"].to("cpu")

  predictedClass_top5=torch.flip(torch.argsort(torch.softmax(model_(input_ids,attention_mask,input_ids_,attention_mask_),axis=-1),dim=-1),dims=[-1])[:,:5]
  predictedScores_top5=torch.softmax(model_(input_ids,attention_mask,input_ids_,attention_mask_),axis=-1)
  answer={}
  for x in range(5):
      answer[model1_GroundTruth[predictedClass_top5[:,x]]]=predictedScores_top5[:,predictedClass_top5[:,x]]
  return answer


if __name__ == '__main__':
  print(get_scores("Sex is integral to human life"))