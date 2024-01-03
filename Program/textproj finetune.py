import os
from PIL import Image
import numpy as np
import clip
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from torchvision.datasets import MNIST
mnist = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=True)

from loguru import logger

class YourDataset(Dataset):  
    def __init__(self,dataset,preprocess,sam_num):

        self.img_process = preprocess

        self.samples = []
        self.sam_labels = []
        self.samid=[]

        catcount=[0,0,0,0,0,0,0,0,0,0]
        for image,cl_id in dataset:

            if(catcount[cl_id]<sam_num):
                label=dataset.classes[cl_id]
                label = "a photo of " + label
                self.samples.append(image)
                self.sam_labels.append(label)
                self.samid.append(cl_id)
                catcount[cl_id]+=1
            if(all(item >sam_num-1 for item in catcount)):
                break
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.samples[idx]
        token = self.tokens[idx]

        image = self.img_process(image)
        return image,token,self.samid[idx]

test_set= MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False)

sam_numo=[1,5,10,15,20,25,30,40,60]

finalresult=[]



def mytrain(sam_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, preprocess = clip.load("ViT-B/32",device=device,jit=False)
    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam([net.text_projection], lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    datasets=YourDataset(mnist,preprocess,sam_num)
    your_dataloader=DataLoader(dataset=datasets,batch_size=10,shuffle=False,num_workers=4,pin_memory=False)
    total_length=sam_num*10
    
    phase = "train"
    model_name = "your model name"
    ckt_gap = 4
    epoches = 30

    net.train()


    for epoch in range(epoches):

        total_loss = 0
        batch_num = 0
        with torch.cuda.amp.autocast(enabled=True):
            for images,labels,cl_id in your_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = net(images, labels)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss

                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(net) 

                batch_num+=1
            epoch_loss = total_loss / total_length
            logger.info('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            
    net.eval()    
    count=0
    datasets=test_set
    for i in range(len(datasets.targets)):
        image,cl_id=datasets[i]
        image_input = preprocess(image).unsqueeze(0).to(device)

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in datasets.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = net.encode_image(image_input)
            text_features = net.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        if(i%500==0):
            print(f"time:{i} actual class of {cl_id}:",datasets.classes[cl_id])
            print("count:",count)
        if(indices==cl_id):
            count+=1
    
    print("最终精确度：",count/len(datasets.targets))
    finalresult.append(count/len(datasets.targets))



for i in sam_numo:
    mytrain(i)
    
    
name=(os.path.basename(__file__).split('.'))[0]

f=open(str(name)+".txt","w")
f.writelines(str(finalresult))
f.close()
