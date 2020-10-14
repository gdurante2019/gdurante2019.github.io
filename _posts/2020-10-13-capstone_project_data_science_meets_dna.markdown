---
layout: post
title:      "Capstone Project:  Data Science meets DNA (Part 1 of 2)"
date:       2020-10-13 15:37:21 -0400
permalink:  capstone_project_data_science_meets_dna
---


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


