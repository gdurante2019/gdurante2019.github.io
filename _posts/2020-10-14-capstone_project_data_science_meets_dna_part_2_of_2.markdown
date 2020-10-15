---
layout: post
title:      "Capstone Project: Data Science meets DNA (Part 2 of 2)"
date:       2020-10-14 21:12:28 -0400
permalink:  capstone_project_data_science_meets_dna_part_2_of_2
---

## Recap of Part 1
In Part 1 of my capstone project blog, I provided an overview of my capstone project, based on the Genetic Engineering Attribution Challenge, hosted by DrivenData for altLabs.  I gave a brief primer on DNA and plasmids, and outlined the problem that altLabs is seeking to address (the ability to identify the lab-of-origin just by the DNA sequence and some binary features of the plasmids).  I followed this up with data exploration, feature engineering, and initial modeling using Random Forests, and commented on the performance of the Random Forest models.  Finally, I laid out a thought process for evaluating which additional modeling approaches might be useful given the unique characteristics of DNA sequences relative to language or image processing, providing a rationale for using a 1D Convolutional Neural Network (CNN) for the next phase of modeling.  

In Part 2 of my capstone project blog post, I will pick up on the thought process for selecting 1D Convolutional Neural Networks (CNNs) for this project, describe my data preprocessing steps, discuss my results, and identify next steps and potential future work.  

## Capstone Project:  Problem Statement
The purpose of the competition is to encourage participants to develop algorithms that can predict with reasonable accuracy which lab produced a particular DNA plasmid.  

## Key Considerations for Selecting a Particular Neural Network (NN) Architecture 
As I described in my last post, a big ‘a-ha’ moment for me was realizing the ways in which DNA sequences share some characteristics with language, and other characteristics with images, but also have some characteristics that are very different from these (especially language).  Having a certain intuitive understanding of what NNs can and can’t do effectively will likely help improve model setup and performance.  

## CNN architecture

While NNs possess formidable computational complexity, the basic structure of CNNs at a high level can be relatively easy to understand.  While I am still at the early stages of developing an intuition for what different NN architectures can accomplish, I did want to give a sense for what is happening conceptually in a CNN:

* Each convolutional layer of a CNN contains ‘filters’ that each scan an entire image (or text sequence) and produce feature maps containing localized information within the image 
* The more filters per convolutional layer, the more features get extracted and the better the model does at recognizing patterns in new unseen images or text
* Pooling layers reduce the number of dimensions and help the model to recognize patterns among features–even distant ones
* Certain methods can be employed to reduce overfitting, such as adding “dropout” layers, in which a random proportion of neurons is turned off, resulting in some information loss but also a reduced tendency for overfitting (dropout layers are used only during training, not validation)
* For classification tasks (such as assigning a probability to a lab producing a certain plasmid), the last part of the model is a Multilayer Perceptron, which has dense connections between the inputs and subsequent layers

An simple schematic gives an understanding about how convolutional layer filters work:

![Simple_conv_layer.gif](https://github.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/blob/master/simple_CNN_layer_filter.gif)

*.gif file from https://medium.com/syncedreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-42f33d4378e0*




Another slightly more complicated schematic provides some intuition about how convolutional layers develop feature maps that the model can then use to extract meaningful information:

![Intuitive_CNN.gif](https://github.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/blob/master/intuitive_CNN.gif) 

*.gif file from KD Nuggets article “An Intuitive Explanation of Convolutional Neural Networks”,  https://www.kdnuggets.com/2016/11/intuitive-explanation-convolutional-neural-networks.html/2*



## Preprocessing Data: Reducing Data Set Size
My initial modeling attempts were painfully slow, due to the size of the data set and the nature of CNNs.  I realized that I needed to reduce the size of the data set so that I could at least get a minimally viable model running.  I employed two different methods to reduce the size of the data set being fed into the model:
1.	Set a maximum character length for DNA sequences (e.g., 5000 base pairs (bp), 8000 bp)
2.	Select a smaller subset of plasmids according to some criterion

For the CNN model runs in this phase of the project, I selected all sequences submitted by labs contributing at least 200 plasmids to the database.  The reason I chose this number is because while the dataset is still large, it contains about half of the data points as the original dataset.  Further rather than 1,314 targets, this reduced data frame contains only 42 labs.  Although the model would have to be refined if I decide to pursue further analysis for the competition to include the entire dataset, using a smaller dataset made it easier to get the model ‘off the ground’, so to speak, allowing for more rapid iteration and modification of parameters to try to improve performance. 

The first approach—setting the maximum character length for a sequence—is quite straightforward using keras’ ```pad_sequences``` and setting the ```maxlen``` argument to whatever the sequence length I wanted to test.  The downside to setting a maximum character length is that you lose the information in that sequence—information which could be helpful to ascertain the lab-of-origin for that plasmid.  

The second approach—reducing the number of data points in the data set—involved writing a function to generate the data subset to use in the model.  The function has the following steps:
1.	Take in training data frame that includes both training values and training labels (lab IDs) and the minimum number of plasmids (‘n’) a lab must have contributed to be included in the smaller data set
2.	Group labs by number of sequences contributed to database, sorted from highest to lowest
3.	Select only the labs that have contributed at least ‘n’ plasmids 
4.	Convert resulting series of labs contributing ‘n’ or more plasmids into a list
5.	Feed that list into the ```.isin()``` method on a copy of the original data frame to obtain all plasmids and associated features associated with each of the labs on the list
6.	Return 3 data frames with this subset of labs and sequences:
a.	Training data set (values and lab IDs) 
b.	Training values data frame
c.	Training labels data frame

Here’s what the code looks like:

```
def reduce_df_size(df, num_plasmid):

    df = df.copy()
    
    labs_grouped = df.groupby(['lab_id']).count().sort_values(by='sequence', 
                                                              ascending=False)    
    labs_grpd_num_plasmid = labs_grouped.loc[labs_grouped['sequence'] >= num_plasmid]
    labs_grpd_num_plasmid = labs_grpd_num_plasmid.reset_index()
    df_subset_list = labs_grpd_num_plasmid.lab_id.tolist()
    df_subset = df.loc[df['lab_id'].isin(df_subset_list)]
    print(f"There are {len(df_subset_list)} labs in this subset dataframe")
    print(f"Each lab has submitted at least {num_plasmid} plasmids")
    df_subset_labs_seqs = df_subset[['sequence', 'seq_len', 'lab_id']]
    train_seqs_subset = df_subset_labs_seqs.drop(['seq_len','lab_id'], axis=1)
    train_labs_subset = df_subset_labs_seqs.drop(['sequence', 'seq_len'], axis=1)
		
    return df_subset_labs_seqs, train_seqs_subset, train_labs_subset
		    
```


## Preparing the data for 1D Convolutional Neural Network (CNN) modeling
### Tokenizing / one-hot encoding values and labels

I used the keras library for python to create the CNNs.  All data must be encoded prior to analysis with NNs, either one-hot encoded, or as integers (requiring the use of an embedding layer as the first/input layer).  For DNA sequences, I used keras’ Tokenizer with ```char_level=True``` and ensuring that the first layer of the CNN was set up as an embedding layer.  Lab ID labels were one-hot encoded.  

### Building the 1D CNN model

Based on my research into CNNs and RNNs, I decided to use the following architecture:  
* Embedding input layer
* Two stacks of the following:
    * Conv1D layer (ReLu Activation)
    * MaxPooling layer (pool_size = 2)
    * Dropout layer (0.2, or 20%) (dropout layer was added starting with the 2nd iteration of the model)
* Flattening layer (to enable outputs of previous layers to be fed into a Multilayer Perceptron NN)
* One or two dense layers (depending on what I was testing) with ReLu Activation
* An output layer (activation = ‘softmax”; returns classifications as probabilities)


## Results and Discussion
### Effects of architecture and hyperparameter tuning on model performance

A few comments on model architecture and hyperparameter tuning:
* While time and computing resources limited my experimentation with different architectures, I did make a few adjustments in terms of layer architecture and hyperparameter tuning
  * Reducing the dimensionality in the layers (e.g., reducing filter size and/or kernel size) helped the model to run faster, while also giving results that were just as good if not slightly better than larger models
  * Adding a second convolutional layer to an existing stack of a single 1D convolutional layer/maxpooling layer/dropout layer didn't seem to give better results
    * I had added a second Conv1D layer in each stack of Conv1D/MaxPooling/Dropout layers and reduced kernel size because I had read that this speed up the model while giving similar or possible better results
    * However, it did not seem to improve model performance in the limited number of runs that I did with this extra Conv1D layer

### Findings across models

Looking across all of the results for model_1 through model_6, I'm struck by how similar the outcomes are, even after modifying some of the hyperparameters, layer structure (adding dropout layers, adding additional Conv1D layers), and trying different maximum character lengths. 

A couple of possible explanations for this outcome spring to mind:
* First, getting an obvious one out of the way: maybe the modifications made to the model don't have much of an impact on the overall performance for this particular problem
* Second: setting a max_char limit to DNA sequence length probably reducing model prediction somewhat
  * DNA plasmids can contain important information towards the end of the sequence
  * While the percentage of sequences exceeding the maximum sequence length was relatively small, the information left out of the analysis from those long sequences might well contain useful and possibly unique information that could help prediction 
* Third, and perhaps the most important explanation of the three: Having such a small percentage of labs represented most likely hinders the training process
  * The model only sees plasmids from labs that have submitted a lot of plasmids, and has no information at all about the plasmids from labs that contribute smaller numbers of plasmids; thus, the model does not have the benefit of guidance in the form of mapping between a great many plasmids and the labs that created them
  * Labs contributing large numbers of plasmids are likely to be large labs with many projects going on, so there is likely to be a lot of diversity within each lab’s portfolio
  * Furthermore, labs with large portfolios are likely to have significant overlap in terms of the DNA sequences, such as promoter regions, origination sequences, and antibiotic resistance markers, making it even more difficult to resolve where particular plasmids or their building blocks come from
  * Thus, when the only sequences used are from large labs, it is likely that an algorithm will struggle to resolve the lineage or source of the sequence

The more I consider this issue, the more that I realize that if data set reduction is absolutely necessary, it would be better to start with sequences contributed by less prolific labs (in terms of plasmid contribution to the repository from which the data set was built)
* Because smaller contributors are likely to be scientists in university settings with a specific research focus, their sequences are likely to be more specific to the lab and should help the model make those connections more readily
* Plasmids from smaller labs will also convey information about a plasmid’s heritage than will sequences contributed by large labs, many of whom probably obtained the rights to use the plasmids from the more focused research scientists

### An Observation Regarding Data Preprocessing Choices

Having taken statistics courses in the past, and having performed a lot of regression analyses throughout the Flatiron School Data Science program, I have learned several important rules regarding data distribution and modeling approach to ensure statistical validity and accurate results, and I take these rules with me wherever I go.  

For example, if I want to perform, say, a linear regression, and my data isn’t normally distributed, then I know that I have a lot of work ahead of me—either to process the data into a form that will work for the model, or to explore other modeling approaches better-suited to this type of data.  

As another example, this project involves a data set that is severely right skewed in terms of the number of plasmids labs have contributed to the data base.  The vast majority of labs have submitted very small numbers of plasmids; further the number of labs drops precipitously as the number of plasmids contributed grows.  Such skewed data makes me very nervous if I’m doing a regression analysis, and I get anxious thinking about how much work I might have to do to address this issue in order to get valid results.  But while one should not ignore the fact that the data in this case is skewed when using NNs, it doesn’t mean that all is lost.  In fact, the realization I described in the previous section reinforces the point that different modeling approaches can extract information from many different types of data sets.  The key is to know your algorithms:  keep your knowledge fresh about the strengths and weaknesses of each, what they can do and what they can’t.

## Recommendations and Suggested Next Steps

Turning to the modeling results and the recommendation emerging from these results… perhaps the biggest recommendation is that the next steps in this project should involve sequences from smaller labs to see if model performance improves and also whether hyperparameter tuning approaches have a bigger effect on outcomes.

Other suggestions for further analysis include:
* Additional parameter tweaking of the Convolutional 1D neural network described 
* Setting max_char from the end of the sequence rather than the beginning (in other words, for max_char = 5000, taking the last 5000 base pairs rather than the first 5000 base pairs 
* Breaking up a large sequence from a lab into smaller sequences (each one being a separate row but having the same sequence ID and lab ID in the labels) and running those through the model with a max_char limit

Additional modeling approaches to consider:
* Running a Recurrent Neural Network (RNN) using LSTM layers to see how that performance compares to the 1D CNN
* Folding in the binary features along with the sequence data (which has been the sole focus of the neural network portion of this project)


## Closing Comments

I picked this project for my capstone partly because it ties into some of my professional interests, partly because it’s unique, and partly because I figured it would be easier than launching into image recognition, which I have not worked with before (I think I was wrong on that last point!).  The way DNA encodes information is different than either images or language, so I had to put a lot of thought into what DNA’s unique information encoding format means for selecting a modeling approach, and a lot of effort into learning about the kinds of models that would be capable of identifying important sequences in an unsupervised manner and being able to tie specific sequences back to specific labs.

In short, engaging with this rather unusual classification challenge pushed me to expand my knowledge about neural networks in a way that I probably wouldn’t have otherwise.  

