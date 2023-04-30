---
title: "Machine Learning Project 4 Part 2 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

## `Chow-Liu Tree` Results

| Dataset | LL |
|---|---|
| accidents | -33.1881104298622 |
| baudio | -44.374902217296565 |
| bnetflix | -60.25034595063276 |
| jester | -58.226531938334375 |
| kdd | -2.294894071447107 |
| msnbc | -6.540127355063576 |
| nltcs | -6.759044665013505 |
| plants | -16.524015044243548 |
| pumsb_star | -30.807048439578587 |
| tretail | -10.946544781894685 |

## `MIXTURE_CLT` Results

| Dataset | Average(LL) | Standard Deviation(LL) | k |
|---|---|---|---|
| accidents | -33.18811042983879 | 1.0048591735576161e-14 | 2 |
| baudio | -44.37490221728933 | 8.987733679556355e-15 | 20 |
| bnetflix | -60.25034595063605 | 2.912358572177595e-14 | 2 |
| jester | -58.22653193829283 | 1.4210854715202004e-14 | 2 |
| kdd | -2.2948940714524477 | 2.5838329957173783e-13 | 2 |
| msnbc | -6.540127355062833 | 2.7985739530820985e-13 | 10 |
| nltcs | -6.759044665014796 | 1.0812455139964926e-14 | 5 |
| plants | -16.524015044253023 | 1.3293037379376718e-14 | 5 |
| pumsb_star | -30.807048439576494 | 2.2803883632436187e-14 | 2 |
| tretail | -10.946544781902475 | 1.9263441276572677e-14 | 20 |

## `RANDOM_FOREST_CLT` Results

| Dataset | Average(LL) | Standard Deviation(LL) | k | r |
|---|---|---|---|---|
| accidents | -33.10184720769717 | 0.004327426842632239 | 20 | 0.2 |
| baudio | -43.85661295420432 | 0.015816995439991557 | 20 | 0.2 |
| bnetflix | -59.90794804960556 | 0.006209735665339615 | 20 | 0.05 |
| jester | -57.37127218905405 | 0.014483052247614276 | 20 | 0.05 |
| kdd | -2.259588580732936 | 0.0012668202597420357 | 20 | 0.05 |
| msnbc | -6.537946385229766 | 0.0011046978994565836 | 10 | 0.2 |
| nltcs | -6.700325856678383 | 0.008703576798810465 | 5 | 0.1 |
| plants | -16.27803560073059 | 0.027712066933353832 | 20 | 0.05 |
| pumsb_star | -30.686019460283724 | 0.012475708426018148 | 20 | 0.2 |
| tretail | -10.904215343958503 | 0.0025564006493050072 | 20 | 0.1 |

## Questions

Can you rank the algorithms in terms of accuracy (measured using test set LL) based on your experiments?
Comment on why you think the ranking makes sense.

1. `RANDOM_FOREST_CLT`
2. `Chow-Liu Tree`
3. `MIXTURE_CLT`

Mixture CLT and the base CLT perform almost identical, albeit the random initialization
aspect of Mixture CLT causes it to be plus-or-minus some arbitrary value relative to the 
deterministic Chow-Liu tree; however, Mixture CLT performs worse on average in our 
testing set compared to CLT based on Log-Likelihood scores.

The Random Forest approach may be performing better due to the removal of edges from
the mutual information graph (a modification of the Chow-Liu Tree learning step) in order to
diversify the trees output by the algorithm. These trees ensembled together prevent overfitting
of the trees to the train dataset.
