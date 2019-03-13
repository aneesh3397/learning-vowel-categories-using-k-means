# learning-vowel-categories-using-k-means

Children are exposed to a variety of vowels that differ in the way that they sound and in the way that they are produced. How are children able to categorize these sounds into vowel categories? This goal of this project is to see if an unsupervised learning algorithm like k-means can be used to categorize the vowel of English. 

The data used comes from the Peterson/Barney vowel data which consists of f0, f1, f2 and f3 (formants) values for 10 different vowels as produced by 76 speakers of British English. This data can be found in the file voweldata.txt. Typically, the f1 and f2 formants provide enough information to categorize the vowels into their correct identities. The image below plots the vowels on the basis of their f1 and f2 values and are categorized into their identities by color. 

<p align="center">
<img src="https://github.com/aneesh3397/learning-vowel-categories-using-k-means/blob/master/vowels.png" width="750" height="550" img align="center">
</p>

We see that while the vowels do form clusters, there is a non-trivial amount of overlap between them. This is because of the variation between different speakers. Adults tend to have lower pitched vowels than children and male speakers tend to have lower pitched vowels and female speakers. The challenge of clustering vowels into their categories is reflected in the image below, where the vowels have been plotted without separating them into their categories:

<p align="center">
<img src="https://github.com/aneesh3397/learning-vowel-categories-using-k-means/blob/master/vowels_unsegmented.png" width="750" height="550" img align="center">
</p>

The code uses the KMeans() function available in the scikit-learn package. The cluster centers are initialized according to a list of appropriate cluster centers provided by the professor (good_init.txt). The centers can also be assigned randomly. The code can be run from terminal by providing the following arguments: 'new_k_means2.py voweldata.txt 10 good_init.txt'

The model returned the following results (using the good_init.txt):

Precision:  0.38491019145996447

Recall:  0.4078250261415127

F-score:  0.39603642025453556

We see that the k-means algorithm wasn't particularly successful at classifying the vowels into their correct categories. Children likely use more information than just the F1 and F2 formants to determine vowel categories. It is likey that they are sensitive to the fundamental frequency (F0) of each speaker and  might also be making use of visual cues such as the shape of the mouth and determining the gender and age of the speaker. 
