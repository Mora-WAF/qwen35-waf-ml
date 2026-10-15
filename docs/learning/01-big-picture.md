# 1. What are we building?

[Course home](README.md) · Next: [How a model learns](02-how-learning-works.md)

## Start with an ordinary website

A person opens a website, searches for a product, signs in, or uploads a document. Each action sends information to a server. Attackers also send requests, sometimes to access someone else's information or make the application do something it should refuse.

The difficulty is that legitimate activity can look suspicious. A programmer might paste code into a support form. A search query might contain database terms. A busy service might make many requests for a valid reason.

This project explores whether an AI model can use the available evidence to tell these situations apart.

## Two kinds of evidence

For a web request, the model may receive details such as the requested page, submitted fields, and relevant application information. A web application firewall, often shortened to WAF, works with this kind of traffic.

For network activity, the model may receive a summary: how many connections occurred, how long they lasted, or how much data moved. A network intrusion detection system, or NIDS, looks for suspicious behavior in network observations.

A summary gives a different view from the contents of a request. Knowing that a thousand connections occurred does not tell us what each request said. The examples must preserve what was actually available to observe.

## The model has a narrow job

The model chooses a category from a defined list. For example, it might classify an input as ordinary legitimate traffic, an attempt to access an administrative function without permission, or a network scan.

A separate set of rules checks whether that prediction supports a recommendation. Those rules can require application context and a sufficiently reliable confidence score. They can also withhold a decision.

```text
Request details or network observations
                   |
                   v
         Model proposes a category
                   |
                   v
  Rules check evidence and confidence
                   |
                   v
       Offline recommendation or abstention
```

In this repository, recommendations are saved or examined offline. The policy does not itself block live traffic. Connecting such recommendations to a running service would require separate engineering and evaluation.

## The same page can mean different things

Suppose someone requests an administrative report. The application's own records say whether the person has the required role. An administrator reading the report and an ordinary account probing that function can send a request with the same visible address.

![A request interpreted using trusted role information](../assets/minimal-pairs.svg)

The illustration shows why the address alone is insufficient. The permission information must come from the application or another trusted source. A person writing "I am an administrator" in a submitted field does not establish that permission.

## What a useful outcome looks like

A useful model should recognize supported attack patterns while handling legitimate lookalikes. Its evaluation should show where it makes mistakes, which evidence it needs, and how often it cannot support a decision.

As you read the next guides, keep two questions separate: "What category does the model predict?" and "Do we have enough evidence to act on that prediction?"

Continue with [How does a model learn?](02-how-learning-works.md).
