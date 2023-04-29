---
title: "Machine Learning Project 4 Part 2 Report"
author: "Nicholas Baker - ndb180002"
geometry: margin=1in
---

## CLT Results

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

## MIXTURE_CLT Results

| Dataset | Average(LL) | Standard Deviation(LL) | k |
|---|---|---|---|
| accidents | -33.119101809351584 | 1.0048591735576161e-14 | 2 |
| plants | -16.407555344707387 | 2.7519201823675253e-15 | 10 |
| baudio | -44.15088570654333 | 8.987733679556355e-15 | 10 |
| msnbc | -6.53862294554448 | 5.859689391140947e-14 | 5 |
| pumsb_star | -30.79053220491877 | 1.3293037379376718e-14 | 2 |
| nltcs | -6.748035238472859 | 3.7890986362223954e-15 | 10 |
| tretail | -10.830404762849858 | 5.148371239083872e-15 | 10 |
| kdd | -2.1616255338368715 | 3.304692175056504e-14 | 10 |
| jester | -58.00620886385428 | 7.78360568894479e-15 | 20 |
| bnetflix | -60.12546818877335 | 1.0048591735576161e-14 | 2 |


## RANDOM_FOREST_CLT Results

| Dataset | Average(LL) | Standard Deviation(LL) | k | r |
|---|---|---|---|---|

## Questions

Can you rank the algorithms in terms of accuracy (measured using test set LL) based on your experiments?
Comment on why you think the ranking makes sense.