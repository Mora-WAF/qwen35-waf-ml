# Learning from the beginning

This short course explains how this project teaches an AI model to recognize suspicious web and network activity. You can read it without knowing Python, machine learning, or security terminology.

By the end, you should be able to describe what goes into the model, how examples teach it, and what evidence would make its results worth trusting.

## Choose a starting point

| Guide | What you will understand |
|---|---|
| [1. What are we building?](01-big-picture.md) | The problem, the model's job, and the limits of a prediction |
| [2. How does a model learn?](02-how-learning-works.md) | Training, fine-tuning, and using a trained model |
| [3. What makes a useful example?](03-good-examples.md) | Labels, context, realistic examples, and keeping an exam fair |
| [4. How do we judge the results?](04-trusting-results.md) | Missed attacks, false alarms, confidence, and when to withhold a decision |
| [5. Follow one example and practice](05-walkthrough-and-practice.md) | How the parts fit together, with questions and answer explanations |

Read them in order for a complete introduction. If you already understand model training, begin with the guides on examples and results.

## A first map

```text
Prepare useful examples
          |
          v
Teach a model to recognize categories
          |
          v
Test it on examples kept out of teaching
          |
          v
Check whether the evidence supports a recommendation
```

Imagine teaching someone to review a stack of requests. You would show ordinary requests as well as attacks, explain what separates them, and set aside unfamiliar cases for an exam. You would also teach them when the information is too incomplete to support a decision. This project organizes those jobs into a repeatable process.

The situations and numbers used in these guides are teaching examples. They are not measured results from this project's model. For changing dataset counts and experiment results, use the [project progress log](../PLAN.md).

## Words you will meet

| Word | Plain meaning |
|---|---|
| Model | A system that has learned patterns from examples |
| Input | The information given to the model |
| Label | The category an example is supposed to belong to |
| Dataset | A collection of examples |
| Training | Adjusting the model using examples and their expected answers |
| Prediction | The model's proposed answer for an input |
| Benign | Legitimate activity in the situation being examined |
| Policy | Rules that turn a prediction and supporting evidence into a recommendation |

When you want implementation details, continue with the [technical learning path](../README.md#technical-learning-path), [hands-on exercises](../05-hands-on.md), or [full glossary](../glossary.md).
