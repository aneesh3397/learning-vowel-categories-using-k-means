# Unsupervised learning of vowel categories

The learning of a language's phonetic inventory can be a tricky problem for children to figre out, and more so for machines.
Phonemes, which are units of sound (for example the /t/ sound at the begining of 'tree') are identified by their 'place of articulation' and 'manner of articulation'. For more see: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet_chart.
Vowels in particular can be difficult, since compared to consonants, they have less identifying features. 

The data for this project comes from a rather old paper, Peterson & Barney 1952, and consists of measurements of four acoustic features, F0, F1, F2 and F3 values, for two repetitions of 10 different vowels by 76 speakers of British English. There are 1520 data points, each containing the following features; speaker type (male,female,child), speaker number (unique id for each participant), vowel identity (true vowel category) and the aforementioned f0, f1, f2 and f3 formant values. The data looks like this:

<p align="center">
  <img src="https://github.com/aneesh3397/unsupervised-learning-of-vowel-categories/blob/master/Screen%20Shot%202020-01-14%20at%2011.28.01%20AM.png" style="display: block; margin: auto;" height="150" width="350"/>
</p>

