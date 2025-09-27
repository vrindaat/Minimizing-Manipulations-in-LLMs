from transformers import AutoTokenizer,AutoModel
from torch import nn
import pandas as pd
import torch
import numpy as np

def getFrequent(data):
    if data.value_counts().index[0] in [np.nan]:

        return data.value_counts().index[1]
    else:
        return data.value_counts().index[0]


#Visualize the dataset

splits = {'train': 'Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet', 'test': 'Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet'}
df = pd.read_parquet("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/" + splits["train"])


df["finalLabel"] = df.iloc[:, -5:].apply(getFrequent, axis=1)

df=df[["text","finalLabel"]]

df=df.sample(df.shape[0])

len(df["text"].tolist()[:-200])

mappings={ind:df["finalLabel"].unique().tolist().index(ind) for ind in df["finalLabel"].unique()}
df["finalLabel"]=df["finalLabel"].map(mappings)
df["finalLabel"].nunique()


tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model=AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer_=AutoTokenizer.from_pretrained("roberta-base")
model_=AutoModel.from_pretrained("roberta-base")


class customModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=AutoModel.from_pretrained("microsoft/deberta-v3-base")
    self.model1=AutoModel.from_pretrained("roberta-base")
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



batchSize=16

inputIds=[]
attentionMask=[]
count=0
train=df["text"].tolist()[:-700]
for x in range(0,len(train),batchSize):
  currentBatch=train[x:x+batchSize]
  tokenizedBatch=tokenizer(currentBatch,padding="max_length",max_length=256,truncation=True,return_tensors="pt")
  input_ids,attention_mask=tokenizedBatch["input_ids"],tokenizedBatch["attention_mask"]
  inputIds.append(input_ids)
  attentionMask.append(attention_mask)




inputIds_=[]
attentionMask_=[]
count=0
train=df["text"].tolist()[:-700]
for x in range(0,len(train),batchSize):
  currentBatch=train[x:x+batchSize]
  tokenizedBatch=tokenizer_(currentBatch,padding=True,truncation=True,return_tensors="pt")
  input_ids,attention_mask=tokenizedBatch["input_ids"],tokenizedBatch["attention_mask"]
  inputIds_.append(input_ids)
  attentionMask_.append(attention_mask)





#This is to validate the model on the validation step for each epoch

batchSize=16

inputIds_test=[]
attentionMask_test=[]
test=df["text"].tolist()[-700:]
count=0
for x in range(0,len(test),batchSize):
  currentBatch=test[x:x+batchSize]
  tokenizedBatch=tokenizer(currentBatch,padding="max_length",max_length=512,truncation=True,return_tensors="pt")
  input_ids,attention_mask=tokenizedBatch["input_ids"],tokenizedBatch["attention_mask"]
  inputIds_test.append(input_ids)
  attentionMask_test.append(attention_mask)


inputIds_test_=[]
attentionMask_test_=[]
test=df["text"].tolist()[-700:]
count=0
for x in range(0,len(test),batchSize):
  currentBatch=test[x:x+batchSize]
  tokenizedBatch=tokenizer_(currentBatch,padding=True,truncation=True,return_tensors="pt")
  input_ids,attention_mask=tokenizedBatch["input_ids"],tokenizedBatch["attention_mask"]
  inputIds_test_.append(input_ids)
  attentionMask_test_.append(attention_mask)




# Lets create the labels now for training and testing


trainingLabels=torch.tensor(df.iloc[:-700,-1].to_numpy().tolist())
testingLabels=torch.tensor(df.iloc[-700:,-1].to_numpy().tolist())


model=customModel()

def customAccuracy(true,predicted):
  correct,total=0,0
  for item in range(true.shape[0]):
      predictedClass=torch.argmax(torch.unsqueeze(predicted[item],dim=0),axis=1)
      if predictedClass==true[item]:
          correct+=1
      total+=1
  return correct,total



optimizer = torch.optim.Adam(
    [{"params": model.model.parameters(), "lr": 5e-5}, {"params": model.linearLayer.parameters(), "lr": 1e-2},
     {"params": model.finalLayer.parameters(), "lr": 1e-2}], lr=1e-5)
criterion = nn.CrossEntropyLoss()
fullLoss = []
model.to("cuda")
model.train()

for epoch in list(range(20)):
    model.train()
    for batch in range(len(inputIds)):
        input_, attention_ = inputIds[batch], attentionMask[batch]
        input__,attention__=inputIds_[batch],attentionMask_[batch]
        output=model(input_.to("cuda"),attention_.to("cuda"),input__.to("cuda"),attention__.to("cuda"))

        currentLabels = trainingLabels[batch * batchSize:batch * batchSize + batchSize]
        loss = criterion(output, currentLabels.to("cuda"))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    full, numCorrect = 0, 0
    current = np.zeros((1, 6))
    model.eval()
    totalLoss = 0
    with torch.no_grad():
        for batch in range(len(inputIds_test)):
            input_, attention_ = inputIds_test[batch], attentionMask_test[batch]
            input__,attention__=inputIds_test_[batch],attentionMask_test_[batch]
            actualOutput = testingLabels[batch * batchSize:batch * batchSize + batchSize]
            output=model(input_.to("cuda"),attention_.to("cuda"),input__.to("cuda"),attention__.to("cuda"))
            #output = model(input_.to("cuda"), attention_.to("cuda"))

            loss = criterion(output, actualOutput.to("cuda"))
            totalLoss += loss.tolist()
            predictedOutput = nn.Softmax()(output)

            correct, total = customAccuracy(actualOutput, predictedOutput)
            full += total
            numCorrect += correct
        for x in range(len(current)):
            print(f"The accuracy across all the batches is ", numCorrect / full)
    print("The loss across all the batches is ", totalLoss / 30)
    if fullLoss == []:
        fullLoss.append(totalLoss / len(inputIds_test))
        torch.save(model.state_dict(), "model.pt")
    else:
        if totalLoss / len(inputIds_test) < min(fullLoss):
            torch.save(model.state_dict(), "model.pt")
            fullLoss.append(totalLoss / len(inputIds_test))



model.load_state_dict(torch.load("model.pt",weights_only=True))





