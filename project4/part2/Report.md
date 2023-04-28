---
title: "Machine Learning Project 4 Part 2 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

| Dataset | CLT | MIXTURE_CLT | RANDOM_FOREST_CLT |
|---|---|---|
| accidents | -33.1881104298622 | -33.1881104298388 | 
| bnetflix | -60.25034595063276 | -60.25034595063603 | 
| msnbc | -6.540127355063576 | -6.540127355062914 | 
| plants | -16.524015044243548 | -16.524015044253023 | 
| jester | -58.226531938334375 | -58.22653193829282 | 
| baudio | -44.374902217296565 | -44.37490221728937 | 
| nltcs | -6.759044665013505 | -6.7590446650148 | 
| tretail | -10.946544781894685 | -10.946544781902364 | 
| pumsb_star | -30.807048439578587 | -30.807048439576473 | 
| kdd | -2.294894071447107 | -2.2948940714524046 | 

2. Run the EM algorithm until convergence or until 50 iterations whichever is earlier.
See section 3 in [Meila and Jordan, 2001]. Use the following values for k ∈ {2, 5, 10, 20}. 
Test performance using the “test set.”

3. Learn the structure and parameters of the model using the following Random-Forests style approach.
Given two hyper-parameters (k, r), generate k sets of Bootstrap samples and
learn the i-th Tree Bayesian network using the i-th set of the Bootstrap samples
by randomly setting exactly r mutual information scores to 0 (as before use
the Chow-Liu algorithm with r mutual information scores set to 0 to learn the
structure and parameters of the Tree Bayesian network). Select k and r using
the validation set and use 1-Laplace smoothing. You can either set pi = 1/k
for all i or use any reasonable method (reasonable method is extra credit).
Describe your (reasonable) method precisely in your report. Does it improve
over the baseline approach that uses pi = 1/k.

Report Test-set Log-Likelihood (LL) score on the 10 datasets available on MS
Teams. For EM and Random Forests (since they are randomized algorithms), choose
the hyper-parameters (k and r) using the validation set and then run the algorithms
5 times and report the average and standard deviation. Can you rank the algorithms
in terms of accuracy (measured using test set LL) based on your experiments? Comment on why you think the ranking makes sense