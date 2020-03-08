---
layout:     post
title:      "The Longest Valid Parentheses Problem"
subtitle:   "Finding the longest valid parentheses substring"
date:       2020-03-07 12:00
author:     "Jean A. Flaherty"
header-img: "img/longest-valid-parentheses/longest-valid-parentheses-thumbnail.png"
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
    func solveSubproblem(seq: AnySequence<Character>, close: Character) -> Int {
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
        let maxLenInS1 = solveSubproblem(seq: AnySequence(s), close: ")")
        // scanning from the right will solve the problem in S2 as described in the proof
        let maxLenInS2 = solveSubproblem(seq: AnySequence(s.reversed()), close: "(")
        // taking the max of the two results will give us the correct answer
        return max(maxLenInS1, maxLenInS2)
    }
}
```

## Complexity Analysis

- Time complexity: $$O(n)$$. Two traversals of the string.
- Space complexity: $$O(1)$$. Only a handful of extra integer variables.

## Proof

#### Definitions:

|Terminology|Symbol|Meaning|Example|
|-----------|------|-------|-------|
|Invalid Opening Parenthesis|$$[$$|An open parenthesis without a matching closing parenthesis.|$$[()$$|
|Invalid Closing Parenthesis|$$]$$|A closing parenthesis without a matching open parenthesis.|$$()]()$$|
|Subsequence 1|$$S_1$$|The subsequence that precedes the first [ or spans entire sequence if [ is not present.|In $$()]()[(())$$ $$ S_1 = ()]() $$|
|Subsequence 2|$$S_2$$|The subsequence that follows the last ] or spans entire sequence if ] is not present.|In $$()]()[(())$$ $$ S_2 = ()[(()) $$|
|-----------|------|-------|-------|

#### Observation 1:
> Since $$[$$ and $$]$$ are invalid, no valid sequence will contain them. Thus
the problem can be narrow down to searching among the subsequences delimited by
$$[$$ and $$]$$ which are themselves all valid subsequences.

#### Observation 2:
> We do not have to consider the subsequences of a valid sequence because we
are looking for the longest valid sequence.

#### Observation 3:
> $$[$$ never precedes $$]$$ because if it did, the two unmatched parentheses
could be matched with each other and no longer be invalid.

#### Observation 4:
> Because of observation 3 and the fact that $$S_1$$ and $$S_2$$ either share
borders or overlap, $$S_1$$ and $$S_2$$ together span the entire sequence.

#### Observation 5:
> There are no valid subsequences that partially overlap $$S_1$$ and/or $$S_2$$
because the borders of $$S_1$$ and $$S_2$$ are either an invalid parenthesis or
out of sequence. In other words all valid sequences are
subsequences of and fully inside $$S_1$$ and/or $$S_2$$ individually.

#### Observation 6:
> With observation 4 & 5 we can see that the longest valid subsequence will be
in $$S_1$$ and/or $$S_2$$. Thus if we know the max valid length in both $$S_1$$
and $$S_2$$, then the problem for the entire sequence could be solved by taking
the max of those two max lengths.

#### Observation 7:
> By definition $$S_1$$ does not contain $$[$$ and $$S_2$$ does not contain
$$]$$. This fact combined with observation 1 & 2 means that valid sequences
are delimited by $$]$$ in $$S_1$$ and $$[$$ in $$S_2$$ and the max length of
the delimited sequences are the solutions for $$S_1$$ and $$S_2$$ respectively.

#### Lemma - The `solveSubproblem` algorithm solves $$S_1$$ and $$S_2$$ individually:
> `solveSubproblem(seq: AnySequence(s), close: ")")`
scans the sequence left to right incrementing the balance when $$($$ and
decrementing when $$)$$. When balance is < 0 we know that we've encountered an
invalid parenthesis and thus parameters are reset to start the search for the
next valid sequence. When balance is 0 we have a valid sequence so we use the
max op to compare and store the largest valid length so far. When we leave
$$S_1$$ the balance stays > 0 and thus nothing out side of $$S_1$$ is considered.
This is effectively finding the largest length of subsequence delimited by $$]$$
in $$S_1$$ and with observation 7 we know that this solves $$S_1$$.
A symmetric argument can be used to show that
`solveSubproblem(seq: AnySequence(s.reversed()), close: "(")` solves $$S_2$$ by
doing the same thing in reverse. Q.E.D.

#### Proof - The `longestValidParentheses` algorithm solves the problem for the entire sequence:
> The `longestValidParentheses` algorithm solves the problem by first solving
$$S_1$$ and $$S_2$$ with `solveSubproblem` and returning the max of the two results.
By observation 6 this solves the problem for the entire sequence. Q.E.D.
