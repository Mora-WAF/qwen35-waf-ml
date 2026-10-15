# 3. What makes a useful example?

[Course home](README.md) · Previous: [Learning](02-how-learning-works.md) · Next: [Judging results](04-trusting-results.md)

## An example needs a defensible answer

A training example contains information to inspect and an expected category. That category is its label.

Consider a request that contains programming code. Its label depends on what the application does with that code. A developer saving a snippet in a text editor is different from someone trying to make a server execute a command. If the evidence needed to distinguish them is missing, assigning a confident label can teach an unsupported rule.

The project keeps the reasons for labels alongside the examples for review. Those reasons and the answer itself stay out of the input shown to the classifier. Otherwise, the model might learn to copy the answer from the question.

## Examples come from different places

| Kind | Plain meaning | Limit to remember |
|---|---|---|
| Observed | Measurements recorded from an actual source | A recording can still be incomplete or mislabeled. |
| Synthetic | A deliberately constructed situation | It may reflect the creator's habits more than real traffic. |
| Hybrid | Real source material placed in a constructed situation | The constructed parts were not observed in the original setting. |

Each kind can contribute useful evidence. Keeping the origin visible lets us ask whether a good result holds across them. A high score on constructed examples alone cannot establish how the model will behave on a busy website.

## Teach the close calls

Obvious attacks and ordinary requests are a start. Close calls reveal whether the model notices the deciding evidence.

One useful technique is to pair an attack example with a legitimate example that looks similar. For an administrative function, the request may stay the same while the caller's trusted permission changes. For a burst of activity, the distinction might depend on whether the activity is authorized and what effects occurred.

A legitimate example that resembles an attack is called a hard negative. The word "negative" means it does not belong to the attack category. It does not mean the example is poor quality.

## Watch for accidental clues

Suppose every attack example uses `bad.example`, while every legitimate example uses `shop.example`. The model could learn those names and appear accurate without using the security evidence. That accidental clue is a shortcut.

Shortcuts can also come from formatting, missing fields, repeated wording, or which source supplied the record. Mixing the names across both categories helps with the hostname problem. Other clues require their own checks and better examples.

## Keep the exam separate

The dataset has different jobs:

| Collection | Job |
|---|---|
| Training examples | Teach the model. |
| Validation examples | Help choose a version and adjust confidence and decision settings. |
| Test examples | Measure the frozen result after those choices. |

Related examples must stay together. If a request is changed only slightly, placing one version in training and another in test makes the exam easier than it appears. This project groups related records by their shared origin, called lineage, before assigning them to collections.

```text
One original situation
    |-- original request
    |-- encoded variation
    |-- different wording
    `-- paired legitimate example

Keep the related group in one collection.
```

An additional test collection can reserve unfamiliar families or sources where suitable examples exist. It asks a harder question about generalization, though it still covers only the situations actually included.

## Review is part of the work

A file can have the right shape and still contain an incorrect label. Automated checks catch problems such as missing fields, duplicate records, and answers accidentally included in inputs. Review asks whether the evidence really supports the category.

Agreement between AI reviewers is useful for finding some mistakes. Reviewers can share the same misunderstanding, so agreement alone does not establish truth. Disputed or unsupported cases need to be held aside or resolved before becoming trusted teaching material.

Continue with [How do we judge the results?](04-trusting-results.md).
