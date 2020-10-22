# Hussain Kara Fallah
# Assignment #1 - Part of Speech Tagging based on Hidden Markov Models

## How to run:
```
pip install -r requirements.txt
python3 main.py 

use python3 main.py -h to see flag of each dataset
```
## Statement
The task is to implement a HMM based Part-of-Speech Tagger. More specifically, it includes building Hidden Markov Model which is basically an FSM and application of  Viterbi algorithm for decoding given the observation sequences.

## Approach:
Simplifying the model, we consider only the following:
* tag transition 
* word likelihood probabilities 

They are are defined as:

$ P_{transition}(tag_i | tag_{i-1}) = \frac{Count(tag_{i-1},tag_i)}{Count(tag_{i-1})}$

$P_{likelihood}(token_i|tag_i) = \frac{Count(tag_{i},token_i)}{Count(tag_{i})}$


Because the model is purely statistical, the training process is just computation of probability values which is done in linear time. 

Next, having calculated probabilities, we apply Viterbi algorithm on a sequence of tokens in order to predict tags for each word (tag path of predictions). 

Viterbi algorithm is an instance of dynamic programming approach. https://en.wikipedia.org/wiki/Viterbi_algorithm


## Unknown Tokens

There might be the case that some of the tokens has not been seen during the learning process. Which is equivalent to:
$$\forall s \in states :: P_{likelihood}(token|state)=0$$

One of the solutions is to completely ignore this factor while calculating  Following this approach I end up predicting the most probable next tag regardless of the token. 

This solution is better than doing nothing as it increases the chance of correct predictions for the next tokens in a sequence. Otherwise, we would have zeroed the path-probabilities and lose any chance to predict the right tags by marking the rest of the sequence as ‘START’ or any other meaningless tag.

## Datasets

I have used 2 different datasets:
* English Web Treebank
    * [https://catalog.ldc.upenn.edu/LDC2012T13]
    * 254,856 tokens
    * 16,622 sentences
    * blog, email, reviews, social 
* GUM - The Georgetown University Multilayer Corpus
        * [https://corpling.uis.georgetown.edu/gum/]
    * 113,374 tokens
    * 5,961 sentences
    * academic, fiction, news, nonfiction, spoken, web, wiki 
    
## Results


Datasets (including test and train splits) are processed and for each sequence we obtain list of tokens and list of universal Part-of-Speech tags (upostag). I have used lemma of the word for tokenization. 
One of the significant issues in processing is the presence of unknown tokens. It means that some of the tokens have not been seen during the learning, before the predictions.

![](https://i.imgur.com/BuznGBx.png)


As we can see, using 80-20 split is optimal for both datasets. To see what are the most common mistakes, observe the confusion matrix:

![](https://i.imgur.com/iTIpRmi.png)

As we can see, almost every tag is predicted correctly, though there are some fluctuations for Adjectives, Adverbs and some other parts of speech. It’s struggling heavily with proper nouns.


## Examples:

#### Example 1:
Here’s an example where the model failed to predict the tag. The word can was predicted as an auxiliary, but it’s definitely a noun.


Example of model’s mistake.


| tokens    | put your trash can far from your home |
| --------- | ------------------------------------- |
| predicted | VERB PRON NOUN **AUX** ADV ADP PRON NOUN  |
| true      | VERB PRON NOUN **NOUN** ADV ADP PRON NOUN |

#### Example 2:


As we can see from the heatmap above, the model has significant difficulties with proper noun (PROPN) tags, so we observe some of the mistakes:

| tokens    | 275 g (2 cup) plain or all-purpose flour (sift)                                 |
| --------- | ------------------------------------------------------------------------------- |
| predicted | **PROPN** **PROPN** PUNCT NUM NOUN PUNCT ADJ CCONJ ADJ NOUN PUNCT **NUM** PUNCT |
| true      | **NUM** **NOUN** PUNCT NUM NOUN PUNCT ADJ CCONJ ADJ NOUN PUNCT **VERB** PUNCT   |

#### Example 3:

| tokens    | try strawberry, apple, banana, apricot, peach etc.                    |
| --------- | --------------------------------------------------------------------- |
| predicted | VERB NOUN PUNCT **NUM** PUNCT NOUN PUNCT NUM PUNCT **PROPN** X PUNCT  |
| true      | VERB NOUN PUNCT **NOUN** PUNCT NOUN PUNCT NOUN PUNCT **NOUN** X PUNCT |

Nouns are often misunderstood to be proper nouns, which is also reflected in the heatmap (27.8 %). Usually proper nouns have the same semantic features (they act as nouns), which might be a reason they are messed with each other. 


#### Example 4:

Example of model’s correct prediction.

| tokens    | sometimes this information is available, but usually not. |
| --------- | --------------------------------------------------------- |
| predicted | ADV DET NOUN AUX ADJ PUNCT CCONJ ADV PART PUNCT           |
| true      | ADV DET NOUN AUX ADJ PUNCT CCONJ ADV PART PUNCT           |


To conclude, I have used acquired knowledge in NLP to implement an HMM Part-of-Speech tagger using the Viterbi algorithm and made analysis of the model performance on several datasets. In order to assess performance we report sentence and token accuracies and plot confusion matrix. One of the crucial issues is having enough data to train the model.



