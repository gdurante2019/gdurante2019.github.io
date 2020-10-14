---
layout: post
title:      "Capstone Project:  Data Science meets DNA--Part 1 of 2"
date:       2020-10-13 15:37:21 -0400
permalink:  capstone_project_data_science_meets_dna
---


For my Flatiron data science capstone, I chose a unique, engaging, and challenging—*very* challenging, as it turns out—problem posed by a current data science competition (the Genetic Engineering Attribution Challenge, sponsored by altLabs and hosted by DrivenData) to develop new algorithms to predict the labs-of-origin for DNA constructs called plasmids.  Plasmids have been used for decades in molecular cloning applications and are critically important to both research activities and industrial production.  However, the increasing availability of advanced methods and tools for genetic engineering raises the specter of potential harm due to unintended or malicious activities by a broader range of people.  Thus, the development of tools that can correctly identify the lab of origin for a given plasmid is becoming ever more important and urgent.

(As a side note, researching this topic gave me the opportunity to revisit the molecular biology (which I had studied in college) and to work with convolutional neural networks more than I had imagined I would so close to my Flatiron graduation date!)

Before I describe the modeling approaches I took, what I learned, and possible next steps for the analysis, I’ll provide a very brief overview of DNA and plasmids for the uninitiated.

## Quick primer on basic DNA structure and plasmids
![DNA structure.gif](https://github.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/ master/242px-DNA_Structure%2BKey%2BLabelled.pn_NoBB.png)
### DNA Basics
DNA (Deoxyribonucleic acid) is found in all living things and carries the genetic instructions for development, function, growth, and reproduction of each individual.  The basic structure of DNA is the double helix, formed by the attraction of complementary DNA base pairs (A and T, G and C) across a gap between two vertical “backbone” molecules.  The resulting ‘spiral staircase’ structure has the DNA base pairs as steps and the phosphate backbone molecules as the spiraling handrails.  Perhaps even more remarkable than the molecule’s elegant structure is the fact that the four base pairs, or “letters” in this genetic alphabet, are all that are required to to encode the vast majority of genetic and functional information in DNA.  

![DNA_animation.gif](https://raw.githubusercontent.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/ master/DNA_animation.gif)

### What are Plasmids?
While most people have heard of chromosomes, ‘plasmid’ isn’t exactly a household word.  As opposed to chromosomes, which contain large amounts of DNA in a tightly coiled superstructure, plasmids are much smaller circular DNA constructs that contain only the ‘bare bones’ DNA needed to accomplish certain tasks.  Examples of sequences found in plasmids include those for molecular products such as proteins, functional motifs (such as ‘start’ and ‘stop’ signals), attachment sites for macromolecules, regulatory instructions, and more.  Their small size, standardized sequences, and well-characterized performance in various lab conditions make safe, reliable, and replicable research and production activities possible.

## The Business Problem:  Tracking Plasmid Lab-of-Origin
### Why is Plasmid Tracking Important?
While plasmids have been used for decades in research and industrial production (e.g., delivering a gene into a bacterium to produce, say, a medicinal product), recent years have seen an acceleration in the development of methods for ‘reading’ and manipulating DNA.  The more widespread availability and increasingly powerful nature of these tools opens the door to intended or unintended negative consequences.  Thus, the ability to predict the lab or entity that produced a certain plasmid is becoming ever more important.  

*This is the fundamental business problem that the sponsors of the competition are attempting to solve:  improving the ability to correct identify the lab-of-origin of a particular plasmid through the development of algorithms that are capable of this classification.*

### How might one long sequence of A's, T's, G's, and C's tell us who produced it?  
Since DNA sequences don’t contain little flashing neon signs to signify who produced the plasmid, how can the lab-of-origin be identified just from the DNA sequence itself?  Fortunately, the combination of sequences used and the placement of the sequences along the length of the plasmid can provide a kind of rudimentary ‘fingerprint’, revealing information about which lab or labs might have produced it.  

### Challenges with recognizing lab-of-origin by DNA sequence alone
However, recognizing a specific ‘fingerprint’ can be challenging.  Even the relatively simple process of comparing dictionaries of sequences to the sequence of a particular plasmid would be incredibly laborious, and is not reveal the kinds of additional information (such as fine-grained sequence nuances due to lab process variations) that could make the difference in identifying the lab of origin.  

Complicating the matter is the fact that, while there are many, many different types of plasmids, many of these plasmids will have the same sequences providing commonly used ‘workhorse’ elements.  Additionally, labs will often build upon previous well-known plasmid structures, to improve experimental replicability or ensure stability in industrial processes, which can complicate linking the actual producer to a specific plasmid.

Thus, methods more powerful than simply compiling a list of sequences found in a given plasmid are required to not only identify sequences within the plasmid that are important, but to uncover spatial and other relationships across the plasmid that reveal more information about the possible source of the plasmid.  

## Methodology
### Competition Mechanics and Data Sets 
All datasets are provided in .csv format.  The data sets for this project were sourced from AddGene, a non-profit repository of plasmids submitted by scientists from all over the world.  The purpose of the repository is to facilitate sharing of plasmids for scientific research.  

The training values data set contains over 63,000 DNA sequences, along with over 30 binary features representing a number of possible plasmid characteristics.  The training labels dataset contains one-hot encoded labels identifying which one of the 1,314 labs produced each of the 63,017 plasmids.  For competition submissions, a test values data set is provided containing over 18,000 sequences for labeling, along with a .csv file showing the proper submission format.

The competition itself consists of two phases.  For the first phase, rather than asking participants to predict the correct lab out of the entire list of 1,314 labs, they advise participants to format their predictions as a probability distribution of each plasmid coming from each of the 1,314 labs and then run a function that DrivenData provides to collect the top 10 highest probabilities for each sequence and include these in the submission.  If that list of 10 labs with the highest probabilities contains the correct lab for that sequence, that prediction is considered correct and contributes to the model’s overall accuracy.

The DrivenData webpage for this competition provided a link to a blog post by DrivenData that provided some starter code with some very basic data exploration, a simple possible feature engineering approach, a quick random forest analysis, a function for identifying the top 10 most likely labs to have produced each plasmid, and code to ensure that submissions are in the proper format. 

### Beginning the Journey 
To familiarize myself with the data and the project, I walked through the blog and implemented the starter code provided.  Then I explored the data, taking particular note of the variety in the length of sequences (from 20 to 60099 base pairs!), the distribution of sequence lengths in the dataset, and the number of plasmids provided by each lab, which turned out to be heavily skewed, as just a small fraction of all labs contribute the majority of sequences to the AddGene database.  

After exploring the data, I began running models, in a two-phase process described below.

### Modeling Approach:  Phase I—Random Forest
#### Feature engineering and model setup
To help participants walk through a simple example to get predictions and learn how to format them for submission to the competition, the DrivenData blog employed a plain-vanilla Random Forest model with some engineered features (non-repeating permutations of 4 DNA base-pairs from 5 ‘letters’— ‘A’, ‘T’, ‘G’, ‘C’, and ‘N’, a stand-in for any of the other 4, yielding 125 features in total) in addition to the binary features included in the data set.  

I had some ideas for engineered features, so building on this basic modeling approach, I ran both a baseline and two separate models with two different sets of engineered features:  one set contained 3-bp combinations from the 5 letters (including repeats), and the other set contained sequences I collected based on my knowledge that there are some very commonly-used short sequences to accomplish various functions in plasmids.  The second set was by no means exhaustive (containing only around 50 sequences), but the model using this feature set performed better than the baseline, as discussed below.

#### Random Forest model performance
As a reminder, the Random Forest model used here has two scoring metrics.  The first is the accuracy of the model selecting the correct lab out of all possible labs (1,314).  The second is the accuracy of the top-10 most probable labs including the correct lab.  

Performance of Random Forest baseline model:
•	First score (accuracy based on selecting the correct lab out of all labs)
o	The baseline model had an accuracy score of 0.1144770 (11.4%)
o	If we were to just randomly select one lab out of a (very large) hat containing all 1,314 labs (each lab represented once), then 11.4% is much better than chance
o	However, a more realistic random chance evaluation would be to select randomly from the pool of labs where the likelihood of selecting a particular lab were proportional to how many plasmids it contributed to the database; in that case, you would actually be better off just guessing the lab with the largest representation in the database (lab ID 'I7FXTVDP') every time, because that lab contributed over 13% of all plasmids to the database. 
•	Second score (top-10 most likely labs prediction contains the correct lab)
o	Baseline model score: 31% 
o	Again, this is better than the likelihood that the correct lab would be contained in a list of 10 labs chosen at random from the pool of 1,314 (approximately 1 in 131, or 0.76%)
o	However, because 10 labs contribute just over 30% of all plasmids to the database, you wouldn’t do much worse to just picking the top 10 labs by number of contributions for every plasmid

Summary of performance of feature-engineered models:
•	Baseline:  11.4% / 31%
•	DrivenData starter model:  19% / 38% 
•	Model using my first engineered feature set:  17% / 36%
•	Model using my second engineered feature set:  14% / 38%

As we can see, the second, third, and fourth models performed better than baseline, but not by much.  

While it was useful to do some initial modeling based on what was done in the blog, it’s not surprising that Random Forests did not perform particularly well; they’re not able to address spatial information very effectively.  But we know that other modeling approaches, such as neural networks, do a much better job at gleaning features from images and text sequences.  With that, I turned my attention to some of these approaches. 

## Conceptual Approach
### Key considerations for selecting a modeling approach
In thinking about the features of plasmids that could point towards their labs-of-origin, I had a big realization that made my decision about a modeling approach easier:  *DNA sequences share some characteristics with language, and other characteristics with images. * 

#### How are DNA sequences similar to languages?  
I can think of at least two ways:
•	Languages and DNA both use combinations of “letters” in a linear fashion to encode information and convey meaning
•	Furthermore, the combination of words and sequences in a document or plasmid convey information about the author (writing style in the case of language, sequence presence and location for plasmids)

#### How are DNA sequences similar to images?  
•	Images and DNA sequences are similar in that the locations of a sequences relative to each other share some similarities with the local features in images that a neural network analyzes, finding patterns across local features and tying these together into a larger whole
•	DNA sequences can range from extremely short (e.g., just a handful of base pairs, thus essentially a very small localized feature) to very long (e.g., thousands of letters, which can be viewed as a larger feature that stretches across a significant portion of an image)

***There are also some important differences between language structure and DNA structure:***
•	DNA doesn’t have clear ‘punctuation’ and spaces the way languages do, so it can be difficult to tell where important information begins and ends
•	Further, letter substitution (sometimes for multiple base pairs in a row) can occur in certain kinds of DNA sequences, creating a combinatorial headache in which even relatively short sequences of 20-30 base pairs can go from essentially being one ‘word’ to being many acceptable variations of a word

### Ensemble Methods vs. Neural Networks
As I discussed above, DNA sequence structure shares some important similarities with both language structure and image structure.  These similarities are a strong argument for using neural networks.  After evaluating the results of the Random Forest model runs, I decided that my time would be best spent focused on neural networks.  The question was, which neural network model should I start with?  

To the extent that DNA sequences share characteristics with language, a neural network such as a Recurrent Neural Network (RNN) with LSTM or GRU layers could provide beneficial.  On the other hand, the similarities to some aspects of image processing suggest a convolutional neural network (CNN).  

Because of these differences between language and DNA sequences, I figured that some of the tools used for NLP might not work as well in identifying lab-of-origin from plasmid sequences.  I also figured that tools used for image analysis and recognition can do a better job because they can find relationships and structure at a more localized granular level and scale this information up to a meaningful whole.  From the Flatiron School curriculum and my own research, I learned that 1D CNNs are often used in NN analysis involving long sequences of information.  Thus, I decided to proceed first with the CNN and potentially explore RNNs and other approaches if time allowed.


For my Flatiron data science capstone, I chose a challenging, engaging, and fairly unique problem posed by a current data science competition (the Genetic Engineering Attribution Challenge, sponsored by altLabs and hosted by DrivenData) to develop new algorithms to predict the labs-of-origin for DNA constructs called plasmids.  Plasmids have been used for decades in molecular cloning applications and are critically important to both research activities and industrial production.  However, the increasing availability of advanced methods and tools for genetic engineering raises the specter of potential harm due to unintended or malicious activities by a broader range of people.  Thus, the development of tools that can correctly identify the lab of origin for a given plasmid is becoming ever more important and urgent.

(As a side note, researching this topic gave me the opportunity to revisit the molecular biology (which I had studied in college) and to work with convolutional neural networks more than I had imagined I would so close to my Flatiron graduation date!)

Before I describe the modeling approaches I took, what I learned, and possible next steps for the analysis, I’ll provide a very brief overview of DNA and plasmids for the uninitiated.

## Quick primer on basic DNA structure and plasmids
![DNA structure.png](https://github.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/master/242px-DNA_Structure%2BKey%2BLabelled.pn_NoBB.png)
### DNA Basics
DNA (Deoxyribonucleic acid) is found in all living things and carries the genetic instructions for development, function, growth, and reproduction of each individual.  The basic structure of DNA is the double helix, formed by the attraction of complementary DNA base pairs (A and T, G and C) across a gap between two vertical “backbone” molecules.  The resulting ‘spiral staircase’ structure has the DNA base pairs as steps and the phosphate backbone molecules as the spiraling handrails.  Perhaps even more remarkable than the molecule’s elegant structure is the fact that the four base pairs, or “letters” in this genetic alphabet, are all that are required to to encode the vast majority of genetic and functional information in DNA.  

![DNA_animation.gif](https://raw.githubusercontent.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/master/DNA_animation.gif)

### What are Plasmids?
While most people have heard of chromosomes, ‘plasmid’ isn’t exactly a household word.  As opposed to chromosomes, which contain large amounts of DNA in a tightly coiled superstructure, plasmids are much smaller circular DNA constructs that contain only the ‘bare bones’ DNA needed to accomplish certain tasks.  Examples of sequences found in plasmids include those for molecular products such as proteins, functional motifs (such as ‘start’ and ‘stop’ signals), attachment sites for macromolecules, regulatory instructions, and more.  Their small size, standardized sequences, and well-characterized performance in various lab conditions make safe, reliable, and replicable research and production activities possible.

![plasmid_map.png](https://github.com/gdurante2019/dsc-capstone-project-v2-online-ds-sp-000/blob/master/addgene-plasmid-42230-sequence-59271-map.png)

## The Business Problem:  Tracking Plasmid Lab-of-Origin
### Why is Plasmid Tracking Important?
While plasmids have been used for decades in research and industrial production (e.g., delivering a gene into a bacterium to produce, say, a medicinal product), recent years have seen an acceleration in the development of methods for ‘reading’ and manipulating DNA.  The more widespread availability and increasingly powerful nature of these tools opens the door to intended or unintended negative consequences.  Thus, the ability to predict the lab or entity that produced a certain plasmid is becoming ever more important.  

*This is the fundamental business problem that the sponsors of the competition are attempting to solve:  improving the ability to correct identify the lab-of-origin of a particular plasmid through the development of algorithms that are capable of this classification.*

### How might one long sequence of A's, T's, G's, and C's tell us who produced it?  
Since DNA sequences don’t contain little flashing neon signs to signify who produced the plasmid, how can the lab-of-origin be identified just from the DNA sequence itself?  Fortunately, the combination of sequences used and the placement of the sequences along the length of the plasmid can provide a kind of rudimentary ‘fingerprint’, revealing information about which lab or labs might have produced it.  

### Challenges with recognizing lab-of-origin by DNA sequence alone
However, recognizing a specific ‘fingerprint’ can be challenging.  Even the relatively simple process of comparing dictionaries of sequences to the sequence of a particular plasmid would be incredibly laborious, and is not reveal the kinds of additional information (such as fine-grained sequence nuances due to lab process variations) that could make the difference in identifying the lab of origin.  

Complicating the matter is the fact that, while there are many, many different types of plasmids, many of these plasmids will have the same sequences providing commonly used ‘workhorse’ elements.  Additionally, labs will often build upon previous well-known plasmid structures, to improve experimental replicability or ensure stability in industrial processes, which can complicate linking the actual producer to a specific plasmid.

Thus, methods more powerful than simply compiling a list of sequences found in a given plasmid are required to not only identify sequences within the plasmid that are important, but to uncover spatial and other relationships across the plasmid that reveal more information about the possible source of the plasmid.  

## Conceptual Approach
### Key considerations for selecting a modeling approach
In thinking about the features of plasmids that could point towards their labs-of-origin, I had a big realization that made my decision about a modeling approach easier:  *DNA sequences share some characteristics with language, and other characteristics with images. * 

#### How are DNA sequences similar to languages?  
I can think of at least two ways:
* Languages and DNA both use combinations of “letters” in a linear fashion to encode information and convey meaning
* Furthermore, the combination of words and sequences in a document or plasmid convey information about the author (writing style in the case of language, sequence presence and location for plasmids)

#### How are DNA sequences similar to images?  
* Images and DNA sequences are similar in that the locations of a sequences relative to each other share some similarities with the local features in images that a neural network analyzes, finding patterns across local features and tying these together into a larger whole
* DNA sequences can range from extremely short (e.g., just a handful of base pairs, thus essentially a very small localized feature) to very long (e.g., thousands of letters, which can be viewed as a larger feature that stretches across a significant portion of an image)

***There are also some important differences between language structure and DNA structure:***
* DNA doesn’t have clear ‘punctuation’ and spaces the way languages do to clearly denote specific words or phrases, so it can be difficult to tell where important information begins and ends
* Further, letter substitution (sometimes for multiple base pairs in a row) can occur in certain kinds of DNA sequences, creating a combinatorial headache in which even relatively short sequences of 20-30 base pairs can go from essentially being one ‘word’ to being many acceptable variations of a word

## Methodology
### The Data Sets 
The DrivenData webpage for this competition provided a link to a blog post by DrivenData that provided some starter code with some very basic data exploration, a simple possible feature engineering approach, a quick random forest analysis, a function for identifying the top 10 most likely labs to have produced each plasmid, and code to ensure that submissions are in the proper format. 

All datasets are provided in .csv format.  The training values data set contains over 63,000 DNA sequences, along with over 30 binary features representing a number of possible plasmid characteristics.  The training labels dataset contains one-hot encoded labels identifying which one of the 1,314 labs produced each of the 63,017 plasmids.  For competition submissions, a test values data set is provided, along with a .csv file showing the proper submission format.

### Beginning the Journey 
To familiarize myself with the data and the project, I walked through the blog and implemented the starter code provided.  Then, I went about engineering some features from the DNA sequences provided, and also ran these through a basic Random Forest model to see if the features I engineered were an improvement on the base case provided in the blog.  Afterwards, I spent some time researching efforts and advances made in this field to date and refined my approach.

### Ensemble Methods vs. Neural Networks
As I discussed above, DNA sequence structure shares some important similarities with both language structure and image structure.  These similarities are a strong argument for using neural networks.  After evaluating the results of the Random Forest model runs, I decided that my time would be best spent focused on neural networks.  The question was, which neural network model should I start with?  

To the extent that DNA sequences share characteristics with language, a neural network such as a Recurrent Neural Network (RNN) with LSTM or GRU layers could provide beneficial.  On the other hand, the similarities to some aspects of image processing suggest a convolutional neural network (CNN).  

Because of these differences between language and DNA sequences, I figured that some of the tools used for NLP might not work as well in identifying lab-of-origin from plasmid sequences.  I also figured that tools used for image analysis and recognition can do a better job because they can find relationships and structure at a more localized granular level and scale this information up to a meaningful whole.  Thus, I decided to proceed first with the CNN and potentially explore RNNs and other approaches if time allows.

###### *This blog represents the first of two blog posts on this project.  Check back for the second post, coming soon...*
# 


