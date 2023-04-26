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

Using 50 different initializations over each value of K,
the resulting mean compression ratio for each K is as follows below:

### Koala

| K | Compression Ratio |
|---|---|
| 2  | 7.22 |
| 5  | 4.93 |
| 10 | 4.73 |
| 15 | 4.81 |
| 20 | 4.89 |

### Penguins 

| K | Compression Ratio |
|---|---|
| 2  | 11.39 |
| 5  |  7.41 |
| 10 |  6.58 |
| 15 |  6.55 |
| 20 |  6.47 |
 
* Is there a tradeoff between image quality and degree of compression. What
would be a good value of K for each of the two images?

Yes, there is a decrease in image quality when you increase the amount of
compression (increase the space saved). Compression ratio is highest with
less diversity of pixels (lower K); however we also see that above a certain
K the compression ratio doesn't drop as quickly, so we can still have highly 
detailed images with high compression rates. 

For the **Koala**, ...

For the **Penguins**, ...