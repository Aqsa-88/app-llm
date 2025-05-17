import pandas as pd 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("medical_data.csv")
X= df['Symptoms']
Y = df['Diagonsis']

lable_encoder = LabelEncoder()
y_encoded = lable_encoder.fit_transform(Y)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X).toarray()

X_train , X_test,Y_train , Y_test, = train_test_split(X_vectorized,y_encoded,test_size=0.2)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
Y_train = torch.tensor(Y_train).long()
Y_test = torch.tensor(Y_test).long()


class SimpleMLP (nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SimpleMLP,self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(input_dim,128)
        ,nn.ReLU()
        ,nn.Linear(128,output_dim)
        )
    def forword(self,x):
        return self.fc(x)
    
    model = SimpleMLP(X_train.shape[1],len(lable_encoder.classes_))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


    epochs = 10 
    for epoch in range(epoch):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs,Y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    

    model.eval()
    with torch.no_grad():
        preds = model(X_test.argmax(dim=1))
        acc = (preds == Y_test).float().mean()
        print(f'Test Accuracy: {acc:.4f}')

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path="medical_data.txt",
#     block_size=128
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir="./fine_tuned_model",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=2
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )

# trainer.train()
# trainer.save_model("./fine_tuned_model")
