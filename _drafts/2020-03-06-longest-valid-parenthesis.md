---
layout:     post
title:      "Longest Valid Parenthesis"
subtitle:   "Finding the longest valid parentheses substring"
date:       2020-03-06 10:00
author:     "Jean A. Flaherty"
header-img: "img/longest-valid-parenthesis/longest-valid-parenthesis-thumbnail.png"
category:   algorithms
tags:       [algorithms]
---

## The Problem

This is a problem that can be found on leetcode and is called the
[longest valid parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
problem. The problem is described as follows:

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Example 1:
```yml
Input: "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"
```

Example 2:

```yml
Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"
```

## Solution

Here's a solution implemented in swift:

```swift
class Solution {
    func greedyChoice(seq: AnySequence<Character>, close: Character) -> Int {
        var maxLen = 0, balance = 0, start = -1
        for (idx, char) in seq.enumerated() {
            // decrement balance when we find a close and increment when open
            balance += char == close ? -1 : 1
            if (balance < 0) {
                // reset startIdx and balance because we found an invalid close
                start = idx
                balance = 0
            } else if (balance == 0) {
                // we have a valid sequence so use max to store the largest length so far
                maxLen = max(maxLen, idx - start)
            }
        }
        return maxLen
    }

    func longestValidParentheses(_ s: String) -> Int {
        // scanning from the left will solve the problem in S1 as described in the proof
        let maxLenInS1 = greedyChoice(seq: AnySequence(s), close: ")")
        // scanning from the right will solve the problem in S2 as described in the proof
        let maxLenInS2 = greedyChoice(seq: AnySequence(s.reversed()), close: "(")
        // taking the max of the two results will give us the correct answer
        return max(maxLenInS1, maxLenInS2)
    }
}
```

## Complexity Analysis

- Time complexity: $$O(n)$$. Two traversals of the string.
- Space complexity: $$O(1)$$. Only a handful of extra integer variables

## Proof
