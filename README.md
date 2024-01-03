# ClipProgram

## ⚡️ Abstract
Clip model has shown its power ever since it was released in 2021, and its excellent performance on traditional image classification tasks is one part of it. However, as the original paper of Clip model and our test has both shown, when Clip meets with the test data that are out of the possibility distribution of its original training dataset, even though it's extraordinarily large, the model may still exhibit poor test accuracy. In correspondence to it, we try to dig in the understanding ability of model Clip. And through the technique of finetuning, we research how Clip model responds when it meets with those datasets who have a new distribution of information compared to the original training datasets. Finally, we use Clip to challenge a harder classification task, from this experience we are inspired by new ideas of using models like Clip to construct "World Knowledge", and manually implement a technique of finetuning Clip without altering its parameters.

## 🤖 Coarse-grained finetuning

-   How Clip sees the world (as we increase the sample number)?
    
<p float="left" align="middle">
  <img src="https://github.com/Danny-1-8/ClipProgram/assets/127832063/550b23c4-f7c6-479e-bf50-f936b8ef2da2" width="49%">
</p>

-   How about a much harder task?

    Hateful Memes, constructed by Meta AI, with a much more complex and richer semantic information

