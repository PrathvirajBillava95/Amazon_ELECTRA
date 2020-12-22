---
layout: default
---

# Introduction

## What is ELECTRA and AmazonELECTRA model, and what research problem are we trying to solve?

Natural Language Processing Models have witnessed a considerable improvement lately, especially after the introduction of the Transformer and Attention mechanism in 2017 (Vaswani, et al. 2017). The introduction of BERT was the ImageNet moment for the NLP community. BERT, which stands for Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin, et al. 2018), gained a substantial improvement over its predecessor LSTM and Bi-LSTM. However, since BERT's introduction to NLP, many researchers have started investigating BERT's components and identifying areas where further improvement could happen. One of those improvements is the loss function of BERT model.
The loss function in BERT depends on the idea of masking 15% of the sequence (later 100%) and asking the loss function to predict the original word for the masked sequence through a pre-training process. Those sequences are part of unlabeled unstructured datasets such as Wikipedia and Book Corpus. However, the ELECTRA, Pre-training Text Encoders as Discriminators Rather Than Generators (Clark, et al. 2020), re-constructed the loss function with game theory concepts to be a binary loss function. Instead of predicting the hidden word, in the case of BERT, in ELECTRA, the model will replace one or more word in the sequence with an equivalent word and ask the loss function to predict whether this word is replaced or original. This idea is straightforward but very powerful in capturing the unlabeled dataset's contextual representation, as we can see in figure 1.


# Research Questions

## Can we improve ELECTRA performance in Amazon related tasks by pre-training ELECTRA with Amazon Dataset?

Transformers-based models are based on the idea of pre-training a deep learning model on unlabeled datasets such as Wikipedia. The contextual representation, which is the model's output, depends on the dataset used for pre-training phase, and thus shifting this dataset toward a more specific domain will lead to a domain adaptation process. The reader should not interpret domain adaptation as a particular field of science, such as biology, chemistry, mathematics, computer science, etc. Domain in this context represents a unique characteristic in which corpora(text) forms its vocabulary and structure. For example, the Twitter dataset represents a domain in itself since it has a unique type of vocabulary (informal words mostly) and a short length of sequence which is less than140 characters. The covid-19 dataset (CORD-19) also represents another domain since it uses a specific domain vocabulary in the biomedical field. Promising domain adaptation results in literature such as BioBERT (Lee, et al. 2019), PubMedBERT (TINN, et al. 2020), BioMegaTron (Shin, et al. 2020), and Covid-Twitter (Müller, Salathé and Kummervold 2020) motivated us to study the domain adaptation of Amazon Review Dataset (Jianmo Ni 2019) using the ELECTRA model.

## How is Hyper parameters affecting our model performance?

This sub-question is aimed more toward a general deep learning technique, which is part of our project learning process. To answer this question later, we will investigate trying out different hyperparameters, especially in the fine-tuning phase, such as learning rate, epochs, batch size, FP16 (mixed precision), PyTorch XLA, etc.

## What will be the performance of the AmazonELECTRA model in the out-domain dataset?

Another question that we will be investigating is the performance of AmazonELECTRA with general domain datasets such as GLUE benchmark (Wang, et al. 2018). This question will focus more on whether the potential improvement with AmazonELECTRA is caused by using informal dataset or the domain constrained dataset.

# Methodology

In the case of the ELECTRA model, the model was pre-trained on English Wikipedia + Book Corpus. In our case, we have used Amazon Review Dataset. The dataset is in JSON Format and contains more than 233.1 million reviews covering the period from May 1996 - Oct 2018. We will explain in detail our pre-processing techniques, including both in pre-training and fine-tuning phases.

## Pre-Processing Phase for pre-training

In this phase, we pre-processed the dataset from the JSON dataset to an unlabeled unstructured dataset that contains only reviews, getting rid of other data fields such as overall rating, review ID, etc. Preprocessing also involved splitting each sentence into a new line, so BERT-like models such as ELECTRA could recognize each sentence. This is an essential step to calculate the loss score. (note that there is a sub loss function in BERT-like model called “Next Sentence prediction” NSP that aims to predict the next sentence on the unlabeled dataset). We used the NLTK library from Stanford to do this phase.

## Cleaning and normalizing the dataset

Part of the fine-tuning process is to prepare our dataset from raw amazon review dataset. However, due to copyright issues we cannot share any pre-processed dataset. Instead, we created a script that download the raw dataset from hugging face database which already has “amazon_us_review” dataset. In the experimental side of our project at GitHub we will focus on this dataset till we mange to create scripts for other datasets such as Amazon Sentimental Analysis dataset



This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
