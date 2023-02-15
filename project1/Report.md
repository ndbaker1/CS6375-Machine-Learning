---
title: "Machine Learning Project 1 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

\newpage
# Algorithm Results

---

## enron1

### Multinomial Naive Bayes

* BagOfWords
    * accuracy:       0.9364035087719298
    * precision:      0.9225352112676056
    * recall:         0.8791946308724832
    * f1_score:       0.9003436426116839

### Discrete Naive Bayes

* Bernoulli
    * accuracy:       0.7302631578947368
    * precision:      0.90625
    * recall:         0.19463087248322147
    * f1_score:       0.3204419889502762

### MCAP Logistic Regression

* Bernoulli - learned penalty = 0.001
	* accuracy:       0.9385964912280702
	* precision:      0.9290780141843972
	* recall:         0.8791946308724832
	* f1_score:       0.9034482758620689

* BagOfWords - learned penalty = 0.0001
	* accuracy:       0.9605263157894737
	* precision:      0.9119496855345912
	* recall:         0.9731543624161074
	* f1_score:       0.9415584415584416

### SGD Classifier + GridSearchCV

* Bernoulli
	* accuracy:       0.9627192982456141
	* precision:      0.9177215189873418
	* recall:         0.9731543624161074
	* f1_score:       0.9446254071661238

* BagOfWords
	* accuracy:       0.9627192982456141
	* precision:      0.9342105263157895
	* recall:         0.9530201342281879
	* f1_score:       0.9435215946843853

\newpage
## enron2

### Multinomial Naive Bayes

* BagOfWords
	* accuracy:       0.9435146443514645
	* precision:      0.912
	* recall:         0.8769230769230769
	* f1_score:       0.8941176470588236

### Discrete Naive Bayes

* Bernoulli
	* accuracy:       0.7740585774058577
	* precision:      0.8928571428571429
	* recall:         0.19230769230769232
	* f1_score:       0.31645569620253167

### MCAP Logistic Regression

* Bernoulli - learned penalty = 0.1
	* accuracy:       0.9560669456066946
	* precision:      0.8865248226950354
	* recall:         0.9615384615384616
	* f1_score:       0.9225092250922509

* BagOfWords - learned penalty = 0.0001
	* accuracy:       0.9372384937238494
	* precision:      0.9166666666666666
	* recall:         0.8461538461538461
	* f1_score:       0.8799999999999999

### SGD Classifier + GridSearchCV

* Bernoulli
	* accuracy:       0.9623430962343096
	* precision:      0.9057971014492754
	* recall:         0.9615384615384616
	* f1_score:       0.9328358208955224

* BagOfWords
	* accuracy:       0.9476987447698745
	* precision:      0.9133858267716536
	* recall:         0.8923076923076924
	* f1_score:       0.9027237354085603

\newpage
## enron4

### Multinomial Naive Bayes

* BagOfWords
	* accuracy:       0.9742173112338858
	* precision:      0.9724310776942355
	* recall:         0.9923273657289002
	* f1_score:       0.9822784810126582

### Discrete Naive Bayes

* Bernoulli
	* accuracy:       0.9171270718232044
	* precision:      0.8967889908256881
	* recall:         1.0
	* f1_score:       0.9455864570737605

### MCAP Logistic Regression

* Bernoulli - learned penalty = 0.1
	* accuracy:       0.9705340699815838
	* precision:      0.9629629629629629
	* recall:         0.9974424552429667
	* f1_score:       0.9798994974874371

* BagOfWords - learned penalty = 0.001
	* accuracy:       0.9631675874769797
	* precision:      0.9580246913580247
	* recall:         0.9923273657289002
	* f1_score:       0.9748743718592964

### SGD Classifier + GridSearchCV

* Bernoulli
	* accuracy:       0.9705340699815838
	* precision:      0.9606879606879607
	* recall:         1.0
	* f1_score:       0.9799498746867168

* BagOfWords
	* accuracy:       0.9668508287292817
	* precision:      0.9582309582309583
	* recall:         0.9974424552429667
	* f1_score:       0.9774436090225566

\newpage
# Hyper-parameter Tuning

## LR Penalty (Lambda)

To find the best `lambda` for the LR algorithm I ran tests with different values on a logarithmic scale from `0.0001` to `0.1`.
After training an LR model with each of these parameters using a 70/30 split of the training set,
I would compare their accuracy with the validation portion and take the parameter from the highest scoring model.

## SGDClassifier

The SGDClassifier takes an `alpha` parameter which I tuned using the `GridSearchCV` and passing a logarithmic search space from `0.001` to `10`.
Additionally, we could modify the penalty function to be `l2`, `l1`, `elasticnet`, or simply avoid any penalty at all.

# Questions

1. Which data representation and algorithm combination yields the best performance (measured in terms of the accuracy, precision, recall and F1 score) and why?

	Multinomial Naive Bayes with Bag of Words performs the best, which makes sense because it is more informed than
	Discrete Naive Bayes and has more context sensitivity. LR and SGDClassifier are able to also learn the parameters
	quite well, but they need large data sizes in order to have a better dataset curve to fit.

	For the most if not all measurements, Multinomial Naive Bayes provides higher or equal value compareds to 
	other algorithms. It does not perform perfectly (such as as recall of 1.0), but it has the most consistenly
	high score among the enron4 dataset, which is the largest given for spam/ham.

2. Does Multinomial Naive Bayes perform better (again performance is measured in terms
of the accuracy, precision, recall and F1 score) than LR and SGDClassifier on the Bag
of Words representation? Explain your yes/no answer.

	Yes, In most cases Multinomial Naive Bayes performs as well or slightly better than LR and SGDClassifier.
	LR and SGDClassifier will perform better with larger datasets, but can potentially overfit in smaller datasets 
	of which I believe our spam/ham dataset belongs. Multinomial Naive Bayes is able to make stable predictions even
	from a small size as long as the sample is uniform and well represents the prior biases.

3. Does Discrete Naive Bayes perform better (again performance is measured in terms of
the accuracy, precision, recall and F1 score) than LR and SGDClassifier on the Bernoulli
representation? Explain your yes/no answer.

	No, the Discrete Naive Bayes performs worse than LR and SGDClassifier in almost every case.
	The Discrete Naive Bayes model is insensitive to words appearing more than once,
	which is an important part of the features between spam and real emails.

4. Does your LR implementation outperform the SGDClassifier (again performance is measured in terms of the accuracy, precision, recall and F1 score) or is the difference in
performance minor? Explain your yes/no answer.

	No, my LR implementation was either the same or slightly worse (could have taken more training iterations) than the SGDClassifier.
	Since the SGDClassifier is built upon Stochastic Gradient Ascent while our LR implementation uses
	a Batch Gradient Ascent Approach (as discussed in the corresponding lecture), their outcomes were very similar.
	Interestingly both had higher accuracy on average for the Bernoulli than the Bag of Words model. 

