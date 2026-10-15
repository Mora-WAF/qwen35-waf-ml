# 2. How does a model learn?

[Course home](README.md) · Previous: [Big picture](01-big-picture.md) · Next: [Useful examples](03-good-examples.md)

## Learning from examples

Imagine a trainee reviewing requests with a teacher. The teacher shows a case, gives the expected category, and explains the evidence that supports it. Over many cases, the trainee begins to recognize patterns.

An AI model learns through numerical adjustments. During training, it receives examples with expected answers. A calculation measures how well it predicts the expected answer, and the training process adjusts learned values to improve those predictions. These values are often called weights.

The trainee comparison helps explain the process, but a model does not reason or learn exactly as a person does. It can pick up convenient patterns that happen to work in its examples and fail elsewhere.

## We start with an existing language model

A large language model, or LLM, is trained to work with sequences of text. This project starts with an existing Qwen model and adapts it to a narrower task: read structured security observations and return one category.

That adaptation is called fine-tuning. It uses the model's existing capabilities and gives it practice with the desired inputs and answers. Fine-tuning cannot establish that the examples are correct; that responsibility belongs to the data preparation and review process.

The project uses an adapter: a relatively small set of learned additions alongside the larger base model. The base stays fixed while the adapter is trained. A saved adapter needs its matching base model and configuration to work correctly.

## Teaching and using happen at different times

| During training | During use |
|---|---|
| The example includes an expected answer. | The expected answer is hidden. |
| The process measures prediction error. | The model proposes a category. |
| Learned values are adjusted. | This project's prediction path leaves learned values unchanged. |

Using a trained model is often called inference. The model is not automatically learning from every new request it sees. Improving it later requires a deliberate process for collecting, reviewing, and training on new examples.

## What the model is asked to produce

The input contains observations such as request details or network measurements. The answer is a short category, recorded in a consistent machine-readable format.

During training, the model reads the input as context. The training error is calculated on producing the answer and ending it correctly. This is why the project carefully checks which parts of each example contribute to learning.

The technical name for that error calculation is loss. A lower training loss means the model is getting better at predicting the training answers under that calculation. We still need a separate exam to learn whether it handles unfamiliar examples.

## Why training has limits

A model can become very good at familiar examples while struggling with new ones. This is called overfitting. A student who memorizes last year's exam has a similar problem: the score looks good until the questions change.

More examples help when they add useful, correct variation. Copying one case thousands of times adds repetition without showing many new situations. Wrong labels can also teach the wrong lesson more thoroughly.

Training takes computing time and memory. A GPU is hardware that performs many numerical operations together; the project uses one for training. A short practice run checks whether loading, training, saving, and reloading work before a longer experiment. That check says little about the final model's accuracy.

## A pause to check your understanding

If a model sees a new request during ordinary prediction, should you assume it has learned that request for next time? No. In this project, learning requires a separate training run. Also ask whether a falling training loss proves success on unfamiliar requests; that requires evaluation on examples kept out of training.

Continue with [What makes a useful example?](03-good-examples.md).
