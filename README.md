# ClipProgram

## ⚡️ Abstract
Clip model has shown its power ever since it was released in 2021, and its excellent performance on traditional image classification tasks is one part of it. However, as the original paper of Clip model and our test has both shown, when Clip meets with the test data that are out of the possibility distribution of its original training dataset, even though it's extraordinarily large, the model may still exhibit poor test accuracy. In correspondence to it, we try to dig in the understanding ability of model Clip. And through the technique of finetuning, we research how Clip model responds when it meets with those datasets who have a new distribution of information compared to the original training datasets. Finally, we use Clip to challenge a harder classification task, from this experience we are inspired by new ideas of using models like Clip to construct "World Knowledge", and manually implement a technique of finetuning Clip without altering its parameters.

## 🤖 Coarse-grained finetuning

-   How Clip sees the world (as we increase the sample number)?
    
<p float="left" align="middle">
  <img src="https://github.com/Danny-1-8/ClipProgram/assets/127832063/550b23c4-f7c6-479e-bf50-f936b8ef2da2" width="90%">
</p>

## 🎯 How about a much harder task?

[Hateful Memes](https://ai.meta.com/tools/hatefulmemes/), constructed by Meta AI, with a much more complex and richer semantic information.

<p float="left" align="middle">
  <img src="https://github.com/Danny-1-8/ClipProgram/assets/127832063/abbdec7f-44fe-45d5-9387-79feff8e75ef" width="30%" height="50%">
</p>
    
Seems like the model loses its attention! How should we get Clip aware of the guilt of Hitler? 
We built a finetuning pipeline for it, with both text-vision alignment and text-text alignment.


<p float="left" align="middle">
  <img src="https://github.com/Danny-1-8/ClipProgram/assets/127832063/4e56ff84-c90d-49a0-8db7-ba5c163742e6" width="60%">
</p> 

## 💡 Can you finetune without altering parameters? Why it works?

We apply a tech called [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), which integrate the information of labels into the input image without altering model's parameters, and the result turns out to be better than coarse-grained finetuning. Why this works? 

Just click the [paper](https://github.com/Danny-1-8/ClipProgram/blob/main/ProjectPaper.pdf) and see more!
