# 4. How do we judge the results?

[Course home](README.md) · Previous: [Examples](03-good-examples.md) · Next: [Walkthrough and practice](05-walkthrough-and-practice.md)

## Count both kinds of mistake

A security system can miss an attack or incorrectly flag legitimate activity. Both matter. Missed attacks leave harmful activity unaddressed; false alarms can interrupt users and consume investigation time.

Imagine a fictional test with 1,000 requests. Ten are attacks and 990 are legitimate. A system that calls everything legitimate scores 99% accuracy and misses every attack.

Now imagine a different system detects eight of the attacks and incorrectly flags ten legitimate requests:

| What really happened | Flagged as an attack | Treated as legitimate |
|---|---:|---:|
| Attack | 8 | 2 |
| Legitimate request | 10 | 980 |

It catches 8 out of 10 attacks. But only 8 of its 18 flags are actual attacks. These figures answer different questions:

- Recall asks how many of the real attacks were caught: 80% here.
- Precision asks how many of the flags were right: about 44% here.
- The false alarm rate asks how much legitimate activity was flagged: about 1% here.

These are made-up numbers for learning. In the project, also distinguish a model flag from a final policy recommendation, and count withheld decisions explicitly.

## Confidence needs checking

A confidence score expresses how strongly a prediction is supported under a particular scoring method. If a model assigns about 80% confidence to many predictions, we would want roughly 80% of that group to be correct under comparable conditions.

Checking and adjusting that relationship is called calibration. This project derives scores from the model and fits a calibration setting on examples reserved for that purpose. A model writing "I am 99% sure" would not establish the same evidence.

Even a calibrated score can become unreliable when traffic changes. Evaluation needs to show which kinds of traffic were included, and future use needs checks for changes in those conditions.

## Sometimes the right outcome is to withhold a decision

The policy can return `ABSTAIN`. That means the available prediction and evidence do not support a decision under the rules. It does not mean the request is safe.

Reasons include missing required application information, an unsupported input type, an invalid model answer, or confidence below the chosen cutoff. The tentative model category can still be retained for diagnostics.

The cutoff is usually called a threshold. Raising a confidence threshold generally reduces the number of accepted decisions, leaving more cases without a decision. We need to measure what happens to both false interventions and missed attacks as it changes.

A system that abstains on everything avoids interventions by making no supported decisions. Report its coverage: the share of cases that receive a decision. For attack intervention recall, abstained attacks remain unmitigated.

## Compare with a simpler method

A baseline is a comparison method. For example, a relatively simple text classifier might recognize patterns in request text. If a much larger model cannot improve on that result in the situations that matter, the extra computing cost needs a reason.

Compare methods on the same eligible examples. A rule system that only reads web requests cannot fairly be scored as though it also examined every network case.

## Ask what each result proves

| Evidence | What it supports |
|---|---|
| Data checks pass | The checked data satisfies those checks. |
| A short training run saves and reloads | That recorded training path works for the tested setup. |
| A frozen evaluation reports errors | Model quality on the evaluated examples, with their limits. |
| Representative operational testing | Evidence about suitability for the tested real setting. |

Small tests also leave uncertainty. Seeing no errors in ten examples is weak evidence about rare mistakes. Many variations of one original case provide less independent evidence than the same number of unrelated cases.

When reading a result, ask what examples were tested, whether they influenced earlier choices, how many independent situations they cover, and which errors the headline score hides.

Continue with [Follow one example and practice](05-walkthrough-and-practice.md).
