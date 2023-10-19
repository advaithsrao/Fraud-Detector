  
# Fraud-Detector
Fraud Detection Package with fine-tuned RoBERTa model, equipped with ethical considerations such as Differential Privacy and SMPC

- [Fraud-Detector](#fraud-detector)
  - [Team](#team)
  - [Steps to run](#steps-to-run)
  - [Abstract](#abstract)
  - [Dataset Description](#dataset-description)
  - [Proposed Methodology](#proposed-methodology)
  - [References](#references)

## Team
|Members|
|---|
|Advaith Shyamsunder Rao|
|Falgun Malhotra|
|Hsiao-Chun Hung|
|Vanshita Gupta|

## Steps to run

**All helper functions and run steps can be found here: https://github.com/advaithsrao/Fraud-Detector/wiki/Repository-Helpers**

## Abstract
In today's data-driven landscape, the detection of fraud emails within corporate communications is critical. With email communication still being the most used mode of communication in organizations, hackers over time have found creative ways to bypass several security layers. In 2022 alone, email-based scams have led to losses of over $2.7 billion. 

Over the last few years, transformer-based models have enabled remarkable advancements in natural language understanding, making them a great choice for tasks such as text classification and generation. However, with deeper neural network-based architectures and models pre-trained on huge amounts of data, privacy concerns loom larger, making it imperative to ensure data protection while maintaining the integrity of the analysis.

## Dataset Description
We make use of a rich source of public email communication, the Enron email dataset. In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. The data has been made public and presents a diverse set of email information ranging from internal, marketing emails to spam and fraud attempts. 

In the early 2000s, Leslie Kaelbling at MIT purchased the dataset and noted that, though the dataset contained scam emails, it also had a number of integrity problems. The dataset was updated later, but it becomes key to ensure privacy in the data while we use it to train a deep neural network model.

## Proposed Methodology
We fine-tune a pre-trained RoBERTa model for our fraud detection task. To ensure complete privacy in our dataset, we will experiment and explore techniques such as Differential Privacy, Secure Multi-Party Computation (SMPC), Homomorphic Encryption, Federated Learning, and Data Masking. Through our experiments, we will attempt to investigate these techniques and find the optimal way to ensure the right amount of privacy, without losing out on the performance of our fraud classifier model.

## References
1. Enron Email Dataset - https://www.cs.cmu.edu/~enron/
2. Differential Privacy - https://arxiv.org/abs/1412.7584
3. Secure Multi-Party Computation - https://link.springer.com/article/10.1007/s12525-022-00572-w
4. Homomorphic Encryption - https://arxiv.org/abs/1704.03578
5. Federated Learning - https://arxiv.org/abs/2301.01299
6. RoBERTa Transformer Model - https://huggingface.co/docs/transformers/model_doc/roberta
7. PySyft for Federated Learning - https://towardsdatascience.com/private-ai-federated-learning-with-pysyft-and-pytorch-954a9e4a4d4e#:~:text=PySyft%20is%20an%20open%2Dsource,private%20and%20secure%20Deep%20Learning.
8. Tensorflow Privacy - https://github.com/tensorflow/privacy
9. ML Privacy Meter - https://github.com/privacytrustlab/ml_privacy_meter
10. Microsoft CrypTFlow for SMPC - https://www.microsoft.com/en-us/research/publication/cryptflow-secure-tensorflow-inference/
11. Facebook Crypten for SMPC - https://github.com/facebookresearch/CrypTen
