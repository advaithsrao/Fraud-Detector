# Fraud-Detector
Fraud Detection Package with fine-tuned RoBERTa model, equipped with ethical considerations such as Differential Privacy and SMPC

## TEAM
|TEAM MEMBERS|
|---|
|Advaith Shyamsunder Rao|
|Falgun Malhotra|
|Hsiao-Chun Hung|
|Vanshita Gupta|

## ABSTRACT
In today's data-driven landscape, the detection of fraud emails within corporate communications is critical. With email communication still being the most used mode of communication in organizations, hackers over time have found creative ways to bypass several security layers. In 2022 alone, email-based scams have led to losses of over $2.7 billion. 

Over the last few years, transformer-based models have enabled remarkable advancements in natural language understanding, making them a great choice for tasks such as text classification and generation. However, with deeper neural network-based architectures and models pre-trained on huge amounts of data, privacy concerns loom larger, making it imperative to ensure data protection while maintaining the integrity of the analysis.

## DATASET DESCRIPTION
We will make use of a rich source of public email communication, the Enron email dataset. In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. The data has been made public and presents a diverse set of email information ranging from internal, marketing emails to spam and fraud attempts. 

In the early 2000s, Leslie Kaelbling at MIT purchased the dataset and noted that, though the dataset contained scam emails, it also had a number of integrity problems. The dataset was updated later, but it becomes key to ensure privacy in the data while we use it to train a deep neural network model.

## PROPOSED METHODOLOGY
We will fine-tune a pre-trained RoBERTa model for our fraud detection task. To ensure complete privacy in our dataset, we will experiment and explore techniques such as Differential Privacy, Secure Multi-Party Computation (SMPC), Homomorphic Encryption, Federated Learning, and Data Masking. Through our experiments, we will attempt to investigate these techniques and find the optimal way to ensure the right amount of privacy, without losing out on the performance of our fraud classifier model.

## REFERENCES
[1] Enron Email Dataset
[2] Differential Privacy
[3] Secure Multi-Party Computation  
[3] Homomorphic Encryption
[5] Federated Learning
[6] RoBERTa Transformer Model
[7] PySyft for Federated Learning
[8] Tensorflow Privacy
[9] ML Privacy Meter
[10] Microsoft CrypTFlow for SMPC
[11] Facebook Crypten for SMPC
