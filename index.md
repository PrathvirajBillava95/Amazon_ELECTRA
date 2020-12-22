---
layout: default
---

# Introduction

Natural Language Processing Models have witnessed a considerable improvement lately, especially after the introduction of the [Transformer and Attention mechanism](https://arxiv.org/pdf/1706.03762.pdf) in 2017. The introduction of [BERT](https://arxiv.org/pdf/1810.04805.pdf) was the ImageNet moment for the NLP community. 

[BERT](https://arxiv.org/pdf/1810.04805.pdf), which stands for Pre-training of Deep Bidirectional Transformers for Language Understanding, gained a substantial improvement over its predecessor LSTM and Bi-LSTM. However, since BERT's introduction to NLP, many researchers started investigating BERT's components and identifying areas where further improvement could happen. One of those improvements is the loss function of BERT model. The loss function in BERT depends on the idea of masking 15% of the sequence (later 100%) and then asking the loss function to predict the original word for the masked sequence through a pre-training process. Those sequences are part of unlabeled unstructured datasets such as Wikipedia and Book Corpus. One downside of BERT model is that it only learns from the masked tokens due to which it requires more resources to train the model. Additionally, the mask token only appears in the pre-training stage and not during the finetuning due to which there is a slight loss of performance in fine-tuned models.    

To address this problem in BERT, the [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf) model re-constructed the loss function with game theory concepts, to be a binary loss function. Instead of predicting the hidden word like BERT, the ELECTRA model will replace one or more word in the sequence with an equivalent word and ask the loss function to predict whether this word is replaced or original. This idea is straightforward but very powerful in capturing the unlabeled dataset's contextual representation.

![Architecture](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/ELECTRA_architecture.png)

_ELECTRA Architecture (Source - [ELECTRA paper](https://arxiv.org/pdf/2003.10555.pdf))

In term of results, we can see a noticeable improvement over BERT and other models as shown in the below table

![Table1](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/ELECTRA_GLUE_results.png)

_Results for models on the GLUE test set (Source - [ELECTRA paper](https://arxiv.org/pdf/2003.10555.pdf))

#### Can we further improve ELECTRA's performance in Amazon related tasks by pre-training ELECTRA with Amazon Dataset?

Transformers-based models are based on the idea of pre-training a deep learning model on unlabeled datasets such as Wikipedia. The contextual representation, which is the model's output, depends on the dataset used for pre-training phase, and thus shifting this dataset toward a more specific domain will lead to a domain adaptation process. 

The reader should not interpret domain adaptation as a particular field of science, such as biology, chemistry, mathematics, computer science, etc. Domain in this context represents a unique characteristic in which corpora(text) forms its vocabulary and structure. For example, the Twitter dataset represents a domain in itself since it has a unique type of vocabulary (informal words mostly) and a short length of sequence which is less than 140 characters. The covid-19 dataset ([CORD-19](https://arxiv.org/pdf/2004.10706.pdf)) also represents another domain since it uses a specific domain vocabulary in the biomedical field. 

Promising domain adaptation results in literature such as [BioBERT](https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf), [PubMedBERT](https://arxiv.org/pdf/2007.15779.pdf), [BioMegaTron](https://arxiv.org/pdf/2010.06060.pdf), and [Covid-Twitter-BERT](https://arxiv.org/pdf/2005.07503.pdf) motivated us to study the domain adaptation of [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html) using the ELECTRA model. We call this model as AmazonELECTRA.

Now given our model- AmazonELECTRA, we further try to address the following questions

##### How is Hyper parameters affecting our model performance?

>This sub-question is aimed more toward a general deep learning technique, which is part of our project learning process. To answer this question later, we will investigate trying out different hyperparameters, especially in the fine-tuning phase, such as learning rate, epochs, batch size, FP16 (mixed precision), PyTorch XLA, etc.
>

##### What will be the performance of the AmazonELECTRA model in the out-domain dataset?

>To address this question, we will investigate the performance of AmazonELECTRA with general domain datasets such as [GLUE benchmark](https://arxiv.org/pdf/1804.07461.pdf). This question will focus more on whether the potential improvement with AmazonELECTRA is caused by using informal dataset or the domain constrained dataset.
>


* * *


# Methodology

In the case of the ELECTRA model, the model was pre-trained on English Wikipedia + Book Corpus. In our case, we have used Amazon Review Dataset for pre-training. The dataset is in JSON Format and contains more than 233.1 million reviews covering the period from May 1996 - Oct 2018. We will explain in detail our pre-processing techniques, including both pre-training and fine-tuning phases.

### Pre-Processing Phase for pre-training

In this phase, we pre-processed the dataset from the JSON format to an unlabeled unstructured dataset that contains only reviews, getting rid of other data fields such as overall rating, review ID, etc. Preprocessing also involved splitting each sentence into a new line, so that BERT-like models such as ELECTRA could recognize each sentence. This is an essential step to calculate the loss score. 

> Note that there is a sub loss function in BERT-like model called “Next Sentence prediction” NSP that aims to predict the next sentence on the unlabeled dataset 

We used the NLTK library from Stanford to do this phase.

### Cleaning and normalizing the dataset

Part of the fine-tuning process is to prepare our dataset from raw amazon review dataset. However, due to copyright issues we cannot share any pre-processed dataset. Instead, we created a script that download the raw dataset from hugging face database which already has “amazon_us_review” dataset. In the experimental side of our project at GitHub, we will focus on this dataset till we manage to create scripts for other datasets such as Amazon Sentimental Analysis dataset

### Building the vocabulary file (Features)

The word embedding matrix in a BERT-like model consists of a list of vocabulary and its context-independent representation. Features in this matrix represent the relationship between those words in terms of the context. This is similar to the [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf) model, which tries to capture the word embeddings between the word in an unstructured dataset. We used the following tutorial from Transformers Blog to build our vocabulary file using Google Compute engine with an adequate number of CPUs and RAM volume, 

[Tutorial link](https://huggingface.co/blog/how-to-train)

We generated a specific domain vocabulary (50K words) using Amazon Review Dataset. Below are examples of how word embeddings are constructed inside BERT-like models.

![Figure 2](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/Word_embedding_3.png)

_(Source : [Medium - Understanding BERT-Word Embeddings](https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca))

We choose 50K words like what [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) and [BioMegaTron](https://arxiv.org/pdf/2010.06060.pdf) model used, which seems to be more effective in capturing the contextual representation than the 30K words that was used by BERT and ELECTRA model.

After creating the vocabulary file we start preparing our model for pre-training and fine-tuning, and we will talk about each section in detail separately.

### ELECTRA Hyperparameters 

Hyperparameters in BERT-like models play a significant role in improving the loss score during the pre-training phase and accuracy during the fine-tuning phase. One of these hyperparameters is the batch size. The batch size affects the flexibility of the learning, as stated in [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) paper. The model RoBERTa used up to 8K as batch size, similar to what [PubMedBERT](https://arxiv.org/pdf/2007.15779.pdf) and [BioMegaTron](https://arxiv.org/pdf/2010.06060.pdf) models used. In our experiment, we used up to 1024 as batch size, and we pre-trained our model for more than 250K steps.  The original ELECTRA model was pre-trained using 1M steps with a batch size of 256, which implies both models have an equal number of samples iterated in the pre-training phase.

Moreover, the learning rate is another critical component in the pre-training phase. We kept the learning rate like the one used in the ELECTRA original setting. Learning rate represent the peak value of the learning rate at the 10% steps before decaying to the rest of the pre-training phase. The 10% phase in deep learning is called the warmup stage and it is widely used in literature. After we initiate the pre-training stage with our model, we monitor the progress of the loss score through Tensor Board, as we will show later.

|                    | ELECTRA | AmazonELECTRA |
|:-------------------|:--------|:--------------|
| Model Size         | Base    | Base          |
| Number of layers   | 12      | 12            |
| Hidden Size layers | 768     | 768           |
| Learnin Rate       | 2e^-4   | 2e^-4         |
| Batch Size         | 256     | 1024          |
| Train Steps        | 766K    | 250K          |

_Hyperparameters used for ELECTRA and AmazonELECTRA during pre-training phase.


* * *


# Experiment and results 

### Pre-Training Phase

The first step in this phase is to convert the raw dataset where each sentence is separated with a new line to a tensor dataset. However, this part is time and resource-consuming, so we have used up to 84 CPUs and 400 RAM to complete this process in 1-2 hours. Then later, we uploaded those ~1000 TensorFlow records to our google bucket. We use up to 32 TPUv3 provided by TensorFlow Research Unit TFRC from the Google team. We used ELECTRA opensource project from GitHub for pre-training and finetuning

[ELECTRA GitHub](https://github.com/google-research/electra)

We started the pre-training phase, and we monitor the loss score through the tensor board for 2 days and 18 hours until it reached 250K steps.

![Figure 3](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/Tensorboard_4.png)

_Loss score during the Pre-Training Phase observed using Tensor Board 

### Fine-Tuning Phase

After pretraining the model, the next step was to fine tune it on NLP downstream tasks. We decided to choose two NLP tasks, i.e., Sentence Classification and Question Answering to fine tune our model and analyze the results.

The dataset used for Sentence classification and Question Answering tasks are as follows,

##### Sentence Classification:

``` 
Evaluation metric for this task is Loss score and accuracy.

1. Amazon Sentimental Analysis
  Train Dataset :200k reviews(85Mb), Dev Dataset: 40k reviews (17Mb).
2. Amazon Review Dataset
  Train Dataset: 454k (191Mb), Dev Dataset: 113K (48Mb)
3. SST-2 (GLUE)
  Train Dataset: 67k, Dev Dataset: 1.8k
```

##### Question Answering:

```
Evaluation metric for this task is F1 score and exact match EM

1. AmazonQA (A Review-Based Question Answering Task)
```

Even though we initially decided to fine-tune our models on two tasks, but due to time constraint and resource restriction, we were only able to fine-tune our model for the Sentence Classification task. Question Answering task dataset i.e. AmazonQA is a large dataset (size > 1.5 GB) and requires a lot of preprocessing which needs additional time and resources. So, we are planning to fine-tune the model with AmazonQA dataset in future.

In addition, we also fine-tuned our model with several downstream tasks form GLUE benchmark. Following table lists the downstream tasks from GLUE Benchmark ( Source: [GLUE Benchmark paper](https://arxiv.org/pdf/1804.07461.pdf))

![Figure 4](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/GLUE_tasks.png)

Below table shows the results of AmazonELECTRA model on various NLP downsteam tasks,

![Figure 5](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/AmazonELECTRA_results.png)

We have also used wandb.ai tool from hugging face library which tracks different runs of the model.

![Figure 6](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/wandb.png)

_The effect of fine-tuning AmazonELECTRA for longer steps (epochs) on loss score

The below table shows the results of running Hyperparameters search (grid search) on Amazon US Review Dataset using 3090 RTX

![Figure 7](https://raw.githubusercontent.com/PrathvirajBillava95/slate/master/images/HPO_results.png)


* * *


# Conclusion and Discussion

The first question that we will tackle down in this discussion is whether AmazonELECTRA achieves the excepted results as per our hypothesis or not. As per the results of AmazonELECTRA model on downstream tasks, the Amazon sentimental analysis dataset, Amazon Kaggle dataset, and SST-2 dataset achieved a gain in performance over ELECTRA by 0.5%-0.8%. For the reader, this may not be a noticeable improvement. However, text classification tasks tend to have this marginal performance gain as we can observe from BERT and ELECTRA results. However, we are expecting further improvement in other tasks such as question answering tasks (AmazonQA) and text generation tasks. 

On the other hand, hyperparameters optimization HPO helps deep learning models finding the best points were model coverage. As we can see the results of Hyperparameters search, the gap between the best and worst performance in F1 score is ~3% despite that they all used the same pre-trained model. This highlighted the effect of choosing hyperparameters during fine-tuning phase on the downstream task’s performance. Moreover, this is in line with early findings in literature showing the effect of batch size on BERT performance ([Yao, et al. 2018](https://arxiv.org/pdf/1810.01021.pdf)). 

Finally, one of the interesting points that is debatable here is whether this improvement with AmazonELECTRA is due to the domain constrained dataset or the type of language used on this dataset (informal). We still have some doubts about this point, and with further investigation, we may reach a firm conclusion later.


* * *


# What we have learned so far from this project?

To begin with, we are extremely happy with the experiment's results obtained so far. Even though it is early to commit that our hypothesis is true, we are satisfied with the results obtained and believe that we are on the right track.

ELECTRA is a state of art model in many NLP tasks, working with this model helped us understand the current trends in the NLP domain. Also, it helped us to learn the transformer architecture and differences in the architecture of BERT and ELECTRA. Furthermore, the experiment results from our project illustrated how important it is to use hyperparameters optimization (HPO) to select the optimal hyperparameters for the model. Through this project, we also learned to use handy tools like TensorBoard and wandb which is quite useful to analyze the results. We used Google TPUs for training and finetuning which was quite exciting. TPUs improves the speed of training and finetuning by a significant amount.

On a personal note, we (Prathviraj Billava and Sultan Alrowili) both learned a lot working on this Project. I (Prathviraj) worked on a deep learning project for the first time and since we worked on a SOTA model it was a great learning experience. Also, I (Prathviraj) would like to give special thanks to Sultan Alrowili, who came up with the main idea behind this  project. Working alongside an already experienced person like Sultan helped me understand the concepts quickly and made it easier for me to work on this project

Overall, we explored the behavior of ELECTRA model when pre-trained with a domain specific dataset. Initial analysis of our result says that the model pre-trained with domain specific dataset performs considerably better then the original model trained with generic dataset like Wikipedia. We are further investigating our model by finetuning it with more domain specific datasets as well as generic datasets to strengthen our hypothesis. 

