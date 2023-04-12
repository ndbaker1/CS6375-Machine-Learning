---
title: "Machine Learning Project 4 Part 1 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

* Display the images after data compression using K-means clustering for different values of K (2, 5, 10, 15, 20).

### Koala

![koala-k=2](./images/Koala-2.jpg)
![koala-k=5](./images/Koala-5.jpg)
![koala-k=10](./images/Koala-10.jpg)
![koala-k=15](./images/Koala-15.jpg)
![koala-k=20](./images/Koala-20.jpg)

### Penguins

![penguin-k=2](./images/Penguins-2.jpg)
![penguin-k=5](./images/Penguins-5.jpg)
![penguin-k=10](./images/Penguins-10.jpg)
![penguin-k=15](./images/Penguins-15.jpg)
![penguin-k=20](./images/Penguins-20.jpg)

* What are the compression ratios for different values of K? Note that you have
to repeat the experiment multiple times with different initializations and report
the average as well as variance in the compression ratio.

Using 20 different initializations over each value of K, the resulting mean compression 
ratio for each K is:


| Image | K | Compression Ratio |
|---|---|---|
| koala    | 2  |  0.11 |
| koala    | 5  |  0.19 |
| koala    | 10 |  0.20 |
| koala    | 15 |  0.21 |
| koala    | 20 |  0.21 |
| penguins | 2  |  0.07 |
| penguins | 5  |  0.13 |
| penguins | 10 |  0.14 |
| penguins | 15 |  0.15 |
| penguins | 20 |  0.15 |
 
* Is there a tradeoff between image quality and degree of compression. What
would be a good value of K for each of the two images?

Yes, there is a decrease in quiality when you increase the amount of compression,
but a value of $K=10$ is a high level amount of compressions that still returns an
image which has enough colors to distinguish most, if not all, elements in the image.