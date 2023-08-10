# DeepIMV论文复现（PyTorch版）
学习多组学数据整合时接触到VAE，因而产生复现相关论文的想法，最后找到这篇2021年在AISTATS上发表的名为["A Variational Information Bottleneck Approach to Multi-Omics Data Integration"](https://arxiv.org/abs/2102.03014)的文章。这篇文章采用**变分信息瓶颈**+**PoE**进行模型构建，适用于有监督的多模态缺失场景。原文提供了[相关代码](https://github.com/chl8856/DeepIMV),但是使用TensorFlow1.0实现的，初步运行没有跑通（遇到梯度为nan的情况，阅读代码发现有可能是没有采样reparametrize策略，但不确定），在没有找到其他复现版本的情况下，出于对PyTorch学习的第二目的，本文使用PyTorch对其进行了模型复现，这里提供了从数据爬取、预处理、模型搭建、模型训练及测试的全部代码。  
初步结果显示，预测精度没有达到论文的程度，按照论文的训练思路，较难训练（2023-8-10 22:24:27）。

