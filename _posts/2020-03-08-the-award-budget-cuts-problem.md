---
layout:     post
title:      "The Award Budget Cuts Problem"
subtitle:   "Finding the grant cap that will impact the least number of people"
date:       2020-03-08 12:00
author:     "Jean A. Flaherty"
header-img: "img/algorithms/algorithms-thumbnail.png"
category:   algorithms
tags:       [algorithms]
---

## The Problem

This is a problem that can be found on [pramp](http://pramp.com) and is called
the Award Budget Cuts problem. This is a write up that will describe and prove
a better solution than the one provided by pramp. The problem is described 
as follows:

The awards committee of your alma mater (i.e. your college/university) asked for your assistance with a budget allocation problem they’re facing. Originally, the committee planned to give `N` research grants this year. However, due to spending cutbacks, the budget was reduced to `newBudget` dollars and now they need to reallocate the grants. The committee made a decision that they’d like to impact as few grant recipients as possible by applying a maximum `cap` on all grants. Every grant initially planned to be higher than `cap` will now be exactly `cap` dollars. Grants less or equal to `cap`, obviously, won’t be impacted.

Given an array `grantsArray` of the original grants and the reduced budget `newBudget`, write a function `findGrantsCap` that finds in the most efficient manner a `cap` such that the least number of recipients is impacted and that the new budget constraint is met (i.e. sum of the `N` reallocated grants equals to `newBudget`).

Analyze the time and space complexities of your solution.

#### Example:

```yml
input:  grantsArray = [2, 100, 50, 120, 1000], newBudget = 190

output: 47 # and given this cap the new grants array would be
           # [2, 47, 47, 47, 47]. Notice that the sum of the
           # new grants is indeed 190
```
#### Constraints:

- [time limit] 5000ms
- [input] `grantsArray: Array<Double>`
  * 0 ≤ `grantsArray.length` ≤ 20
  * 0 ≤ `grantsArray[i]`
- [input] `newBudget: Double`
- [output] `Double`

## Solution

Here's a solution implemented in swift:

```swift
import Foundation

func findGrantsCap(grantsArray: [Double], newBudget: Double) -> Double {
  let n = grantsArray.count
  // avoid zero devision
  if n == 0 {
    return 0
  }
  // sort in increasing order - O(n log(n))
  var grantsArray = grantsArray.sorted()
  // affect_count will be the number of grants affected but we'll start out assuming everyone is affected
  var affect_count = Double(n)
  // as we traverse the array of grants we'll update remBudget to reflect the remaining budget
  // so that when we know among how many grants we want to divide the remaining budget
  var remBudget = newBudget
  // we will start by assuming that all grants will be affected and divide the remaining budget equally
  // and if we find the assumption to be wrong, we will keep updating this number
  var dividedRemBudget = remBudget / affect_count
  for i in 0..<n-1 {
    if (grantsArray[i] < dividedRemBudget) {
      // if the current grant is less than our cap lower bound
      // we know that the current grant will not be affected
      affect_count -= 1.0
      // update remBudget to the budget remaining for the remaining grants
      remBudget -= grantsArray[i]
      // recompute lower bound on cap
      dividedRemBudget = remBudget / affect_count
    } else {
      // we have found the actual cap, proof will be provided below
      break
    }
  }
  // after the for loop is done dividedRemBudget == cap
  return dividedRemBudget
}
```

## Complexity Analysis

- Time complexity: $$O(n log(n))$$ due to sorting but could have been $$O(n)$$
for a single traversal if the `grantsArray` was presorted.
- Space complexity: $$O(1)$$. Only a handful of extra integer and double variables.

## Proof

Definitions:

|Symbol|Meaning|Example|
|------|-------|-------|
|$$g_i$$|The i-th smallest grant.|If `grantsArray`$$= [30,20,40]$$, then $$g_2 = 30$$|
|$$r_i$$|The remaining budget when the i smallest grants are budgeted. $$r_i = newBudget - \sum_{j=1}^i(g_j)$$|If `grantsArray`$$= [30,20,40]$$ and `newBudget` $$= 60$$, then $$r_2 = 10$$|
|$$k_i$$|The remaining number of grants to budget when the i smallest grants are budgeted. $$k_i = n - i$$|If `grantsArray`$$= [30,20,40]$$ and `newBudget` $$= 60$$, then $$k_1 = 2$$|
|$$d_i$$|The divided remaining budget when the i smallest grants are budgeted. $$\frac{r_i}{k_i}$$|If `grantsArray`$$= [30,20,40]$$ and `newBudget` $$= 60$$, then $$d_1 = 15$$|
|------|-------|-------|

Observation 1:
> If a grant is unaffected then all smaller or equal grants are also
unaffected. This means there is some threshold where grants get large enough to get capped.
This also means we can greedily select from the smallest grants to determine the
unaffected grants if we know how to determine the threshold.

Observation 2:
> `dividedRemBudget` tells us what the cap would be if the current grant
`grantsArray[i]` was the first grant to get capped thus if `grantsArray[i] < dividedRemBudget`
then `grantsArray[i]` is definitely not getting capped.

Observation 3:
> When `grantsArray[i] < dividedRemBudget` or equivalently $$g_i < d_{i-1}$$
then $$d_{i-1} = \frac{d_{i-1} \cdot (k_{i-1}-1)}{k_{i-1}-1} = \frac{r_{i-1}-d_{i-1}}{k_{i-1}-1} < \frac{r_{i-1}-g_i}{k_{i-1}-1} = \frac{r_i}{k_i} = d_i$$ or more succinctly $$d_{i-1}< d_i$$. By the same argument we can show that when `grantsArray[i] >= dividedRemBudget` or equivalently $$g_i >= d_{i-1}$$ then $$d_{i-1} >= d_i$$.

Observation 4:
> Once `grantsArray[i] >= dividedRemBudget` or equivalently $$g_i >= d_{i-1}$$ then $$d_{i-1} >= d_i$$ for all the
remaining grants because $$g_i$$ keeps increasing and by observation 3 $$d_{i-1}$$
keeps decreasing only making the inequality `grantsArray[i] >= dividedRemBudget` wider or at the least the same.
Therefore there is some threshold where the divided remaining budget $$d_i$$ starts decreasing or staying the same.

Observation 5:
> The first grant where `grantsArray[i] >= dividedRemBudget` is the first grant to get capped (or be equal) to the cap.
If the first to get capped (or be equal) came before this grant this would contradict observation 2.
If the first to get capped (or be equal) came after this grant, the fact that $$d_i$$ will keep decreasing (observation 4)
tells us that our cap will be smaller (or be equal) to `grantsArray[i]` which contradicts our assumption.
Thus the first grant where `grantsArray[i] >= dividedRemBudget` is the first grant to get capped (or be equal) to the cap.
This also means that this is the threshold mentioned in Observation 1.

Proof:
> As the for loop stops when `grantsArray[i] >= dividedRemBudget` at the first grant to get capped (Observation 5), and `dividedRemBudget`$$= d_{i-1}$$. `dividedRemBudget` gives us the cap amount because it is the remaining budget
divided among the $$k_{i-1}$$ affected grants. Q.E.D.
