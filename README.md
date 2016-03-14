# DistilledKnowledge
kaldi-asr-distilled-knowledge

nnet-forward讀取temperature OK的版本

// 修改final.nnet 在softmax後面加上 “<Tempuature> 20” 即可

<<<<<<< HEAD
預計工作：
---
* 進行mtl (blocksoftmax)也能讀取température label --> OK
* nnet proto 可讀取 blocksoftmax 也能讀取température --> OK
* feed-forward 及 back-propagation的溫度--> OK 

目前限制：
---
* nnet-train-frmshuff裡面一定要加溫度的參數（nnet-train-frmshuff --objective-function=multitask,xent,3400,1,10,xent,144,1,20）

