  
# Fraud-Detector
Fraud Detection Package with fine-tuned ML and DL models, equipped with ethical considerations such as Differential Privacy and Homomorphic Encryption

- [Fraud-Detector](#fraud-detector)
  - [Team](#team)
  - [CI/CD Pipeline Status](#cicd-pipeline-status)
  - [Installation and Run Instructions](#installation-and-run-instructions)
  - [Abstract](#abstract)
  - [Dataset Description](#dataset-description)
    - [Data Splits](#data-splits)
    - [Training Label Split](#training-label-split)
  - [DATA ANNOTATION](#data-annotation)
    - [1. AUTOMATED ML LABELING](#1-automated-ml-labeling)
    - [2. EMAIL SIGNALS](#2-email-signals)
    - [3. MANUAL INSPECTION](#3-manual-inspection)
  - [Proposed Methodology](#proposed-methodology)
  - [References](#references)
  - [Citations](#citations)



***

## Team
|Members|
|---|
|Advaith Shyamsunder Rao|
|Falgun Malhotra|
|Hsiao-Chun Hung|
|Vanshita Gupta|

***

## CI/CD Pipeline Status

[![Fraud Detector CI/CD](https://github.com/advaithsrao/Fraud-Detector/actions/workflows/pipeline.yml/badge.svg)](https://github.com/advaithsrao/Fraud-Detector/actions/workflows/pipeline.yml)

***

## Installation and Run Instructions

**All helper functions and run steps can be found in the wiki pages.**

| Helper | Page |
| ------ | ------ |
| Setup Environment and Integrations | [Wiki](https://github.com/advaithsrao/Fraud-Detector/wiki/Setup-Environment) |
| W&B Model tracking and Logging for our experiments | [Wiki](https://github.com/advaithsrao/Fraud-Detector/wiki/Model-Tracking-and-Logs) |
| *How to:* Standalone - Processing and Labeler | [Wiki](https://github.com/advaithsrao/Fraud-Detector/wiki/Load-Preprocessed-and-Labeled-Data) |
| *How to:* Model Training Pipeline | [Wiki](https://github.com/advaithsrao/Fraud-Detector/wiki/Model-Training-Pipeline) |

***

## Abstract
In today's data-driven landscape, the detection of fraud emails within corporate communications is critical. With email communication still being the most used mode of communication in organizations, hackers overtime have found creative ways to bypass several security layers. In 2022 alone, email-based scams have led tolosses of over $2.7 billion.

Over the last few years, Transformer-based models have enabled remarkable advancements in NaturalLanguage Understanding, making them a great choice for tasks such as text classification and generation.However, with deeper neural network-based architectures and models pre-trained on huge amounts of data,privacy concerns loom larger, making it imperative to ensure data protection while maintaining the integrity ofthe analysis.

The goal of the project is to explore Ethics-Driven Machine Learning, building a Fraud Detector model using a pre-trained RoBERTa model, with ethical considerations to the model using techniques such as Differential Privacy,Secure Multi-Party Computation, Federated Learning, and Homomorphic Encryption.

***

## Dataset Description
The project makes use of a rich source of public email communication, the Enron email dataset ( https://www.cs.cmu.edu/~enron/ ). In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. The data has been made public and presents a diverse set of email information ranging from internal, marketing emails to spam and fraud attempts. 

In the early 2000s, Leslie Kaelbling at MIT purchased the dataset and noted that, though the dataset contained scam emails, it also had a number of integrity problems. The dataset was updated later, but it becomes key to ensure privacy in the data while it is used to train a deep neural network model.

Though the Enron Email Dataset contains over 500K emails, one of the problems with the dataset is the availability of labeled frauds in the dataset. Label annotation is done with the goal to detect an umbrella of fraud emails accurately. Since, fraud emails fall into several types such as Phishing, Financial, Romance, Subscription, and Nigerian Prince scams, there has to be multiple heuristics used to effectively label all types of fraudulent emails.

To tackle this problem we use heuristics to label the enron data corpus using email signals as well as perform automated labeling using simple ML models on other smaller email datasets available online. These fraud annotation techniques are discussed in detail in section 4 below.

To perform fraud annotation on enron dataset as well as provide more fraud examples for modeling, the project uses two more fraud data sources:
Phishing Email Dataset: https://www.kaggle.com/dsv/6090437
Social Engineering Dataset: http://aclweb.org/aclwiki

To perform high-quality testing, the project uses two gold label sets:
1. Gold Fraud Set: Contains `1000` curated fraud emails from the phishing and social engineering dataset. On this fraud set, we recall the test, how many fraud emails does our model miss out.
2. Sanity set: Contains 250000 curated internal email communication emails between employees at Enron. On this dataset, we precision test our model to see if it picks up any non-fraud email that it is not supposed to flag as fraud.

Below is a short data summary of the label distribution across different sources (on the x-axis) and labels (on the y-axis).

|  | Fraud | Non-Fraud |
| -- | -- | -- |
| Enron Dataset | 2327 | 445090 |
| Phishing Dataset | 4976 | 12515 |
| Social Engineering Dataset | 4160 | 6475 |


To tackle data imbalance, the project also performs data augmentation, creating 9 synthetic emails for every 1 fraud email. The augmentation process makes use of techniques such as: 
1. Synonym Replacement
2. Stopword Removal
3. Swapping Noun Phrases

### Data Splits

| Set | Emails |
| --- | --- |
| Train | 224543  |
| Sanity | 250000 |
| Gold Fraud | 1000 | 

### Training Label Split

These are the training label splits before annotation

| Label | Emails |
| --- | --- |
| 0 | 214080 |
| 1 | 10463 |


***

## Data Annotation

### 1. Automated ML Labeling

The following heuristics are used to annotate labels for enron email data using the other two data sources:
1. Phishing Model Annotation:  We are annotating mails from the Enron dataset using a high-precision model trained on the Phishing mails dataset. 
2. Social Engineering Model Annotation: We are annotating mails from the Enron dataset using a high-precision model trained on the Social Engineering mails dataset. 

The two ML Annotator models use Term Frequency Inverse Document Frequency (TFIDF) to embed the input text and make use of SVM models with Gaussian Kernel.

### 2. Email Signals

Email Signal based heuristics are used to specifically filter and target suspicious emails for fraud labeling. The signals used are:
Person Of Interest: There is a publicly available list of email addresses of employees who were liable for the massive data leak at Enron. These user mailboxes can have a higher chance of containing quality fraud emails.
1. Suspicious Folders: The Enron data is dumped into several folders for every employee. Folders consist of inbox, deleted_items, junk, calendar, etc. We define a set of folders that have a higher chance of containing fraud emails, such as Deleted Items and Junk.
2. Sender Type: The sender type was categorized as `Internal` and `External` based on their email address. 
3. Low Communication: We defined a threshold of `4` emails on the basis of the table below. A user qualifies as a low-comm sender if their sent mails are less than this threshold. Mails sent from low-comm senders are assigned with a high probability of being a fraud.
4. Contains Replies and Forwards: If an email contains forwards or replies, we assign a low probability of it being a fraud.

The below table represents the distribution of the length of email bodies in terms of words.

| | |
| -- | -- |
| count | 20131 |
| mean | 12.3 |
| std | 104.9 |
| min | 1 |
| 25% | 1 |
| 50% | 1 |
| 75% | 4 |
| max | 5486 |

### 3. Manual Inspection

To ensure high-quality labels, we manually inspect the mismatch examples from ML Annotation to relabel the enron dataset.

***

## Proposed Methodology
We fine-tune ML and DL models for our fraud detection task. To ensure complete privacy in our dataset, we will experiment and explore techniques such as Differential Privacy, Secure Multi-Party Computation (SMPC), Homomorphic Encryption, Federated Learning, and Data Masking. Through our experiments, we will attempt to investigate these techniques and find the optimal way to ensure the right amount of privacy, without losing out on the performance of our fraud classifier model.

***

## References

1. Enron Email Dataset
   - **URL**: [https://www.cs.cmu.edu/~enron/](https://www.cs.cmu.edu/~enron/)

2. Differential Privacy
   - **URL**: [https://arxiv.org/abs/1412.7584](https://arxiv.org/abs/1412.7584)

3. Secure Multi-Party Computation
   - **URL**: [https://link.springer.com/article/10.1007/s12525-022-00572-w](https://link.springer.com/article/10.1007/s12525-022-00572-w)

4. Homomorphic Encryption
   - **URL**: [https://arxiv.org/abs/1704.03578](https://arxiv.org/abs/1704.03578)

5. Federated Learning
   - **URL**: [https://arxiv.org/abs/2301.01299](https://arxiv.org/abs/2301.01299)

6. RoBERTa Transformer Model
   - **URL**: [https://huggingface.co/docs/transformers/model_doc/roberta](https://huggingface.co/docs/transformers/model_doc/roberta)

7. PySyft for Federated Learning
   - **URL**: [https://towardsdatascience.com/private-ai-federated-learning-with-pysyft-and-pytorch-954a9e4a4d4e#:~:text=PySyft%20is%20an%20open%2Dsource,private%20and%20secure%20Deep%20Learning.](https://towardsdatascience.com/private-ai-federated-learning-with-pysyft-and-pytorch-954a9e4a4d4e#:~:text=PySyft%20is%20an%20open%2Dsource,private%20and%20secure%20Deep%20Learning.)

8. Tensorflow Privacy
   - **URL**: [https://github.com/tensorflow/privacy](https://github.com/tensorflow/privacy)

9. ML Privacy Meter
   - **URL**: [https://github.com/privacytrustlab/ml_privacy_meter](https://github.com/privacytrustlab/ml_privacy_meter)

10. Microsoft CrypTFlow for SMPC
    - **URL**: [https://www.microsoft.com/en-us/research/publication/cryptflow-secure-tensorflow-inference/](https://www.microsoft.com/en-us/research/publication/cryptflow-secure-tensorflow-inference/)

11. Facebook Crypten for SMPC
    - **URL**: [https://github.com/facebookresearch/CrypTen](https://github.com/facebookresearch/CrypTen)

12. Phishing Dataset
    - **URL**: [https://www.kaggle.com/dsv/6090437](https://www.kaggle.com/dsv/6090437)

13. Social Engineering Dataset
    - **URL**: [https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset?rvi=1](https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset?rvi=1)

## Citations

**`Phishing Email Detection Dataset`**

- **Title**: Phishing Email Detection
- **URL**: [https://www.kaggle.com/dsv/6090437](https://www.kaggle.com/dsv/6090437)
- **DOI**: 10.34740/KAGGLE/DSV/6090437
- **Publisher**: Kaggle
- **Author**: Subhadeep Chakraborty
- **Year**: 2023

**`CLAIR Fraud Email Collection`**
- **Title**: CLAIR collection of fraud email
- **URL**: [http://aclweb.org/aclwiki](http://aclweb.org/aclwiki)
- **Author**: Radev, D.
- **Year**: 2008
