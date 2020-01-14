# Unsupervised learning of vowel categories

The learning of a language's phonetic inventory can be a tricky problem for children to figre out, and more so for machines.
Phonemes, which are units of sound (for example the /t/ sound at the begining of 'tree') are identified by their 'place of articulation' and 'manner of articulation'. For more see: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet_chart.
Vowels in particular can be difficult, since compared to consonants, they have less identifying features. 

The data for this project comes from a rather old paper, Peterson & Barney 1952, and consists of measurements of four acoustic features, F0, F1, F2 and F3 values, for two repetitions of 10 different vowels by 76 speakers of British English. There are 1520 data points, each containing the following features; speaker type (male,female,child), speaker number (unique id for each participant), vowel identity (true vowel category) and the aforementioned f0, f1, f2 and f3 formant values. The data looks like this:

<p align="center">
  <img src="https://github.com/aneesh3397/unsupervised-learning-of-vowel-categories/blob/master/header.png" style="display: block; margin: auto;" height="150" width="350"/>
</p>

Our task now is to use the available features to predict the true vowel category for each data point. We impose the constraint of using unsupervised methods to achieve this to simulate the problem as it exists for children; they don't get feedback (labeled data) on whether or not they have correctly identified a vowel. In fact, a simple logistic regression model can correctly classify data points with about 70% accuracy, using just the f1 and f2 formant values:

```python
# get f1 and f2 format values:
X = []
for index, row in df.iterrows():
    X.append([row['f1'],row['f2']])
```
```python
# get true vowel ids:
y = df['vowel_id'].values
```
```python
# create 80-20 train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```python
# fit model and score on test data:
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)
```
The model scores 70.0657%. Incorporating f0 and f3 formant values increases this to about 84%. 

### Unsupervised clustering:

Before we attempt to cluster these data points into our vowel categories, let's see what the data looks like in a 2d space, plotting the f1 and f2 values:

<iframe src="vowel_2d.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="500"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe>




