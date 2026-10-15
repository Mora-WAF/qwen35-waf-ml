# Glossary

[Learning index](README.md)

| Term | Meaning in this project |
|---|---|
| Adapter | Small learned parameter set loaded alongside the pinned base model |
| Abstention | Policy outcome that withholds a supported class/action decision |
| Baseline | Simpler comparison method, such as text SVM or HTTP rule matching |
| Benign twin | Legitimate example paired with an attack scenario and sharing its lineage |
| Calibration | Fitting how model scores become probabilities on a separate partition |
| Checkpoint | Training state needed for continuation, beyond an inference adapter |
| Confuser set | Neighboring classes with similar surface form and different deciding evidence |
| Context | Application/environment evidence; it must come from a trusted source |
| Core | Records belonging to the supervised 41-class task |
| Coverage | Fraction of examples receiving a non-abstain policy decision |
| EOS | End-of-sequence/end-of-turn token supervised with the answer |
| False intervention | A benign record receiving BLOCK, RATE_LIMIT, or CHALLENGE |
| Fingerprint | Recorded tokenizer/template/prompt identity used to detect drift |
| Gate | Validator whose failure prevents claiming data readiness or normal packaging |
| Hard negative | Legitimate content or behavior that resembles an attack |
| Hybrid | Real seed evidence combined with generated context |
| JSONL | One JSON record per line |
| Lineage group | Related examples that must remain together in one split |
| Loss mask | Token labels selecting which output positions contribute to training loss |
| Minimal pair | Examples differing in a small amount of deciding evidence |
| Observation boundary | The point/window up to which a feature could be known |
| Provenance | Whether evidence is observed, synthetic, or hybrid, with source details |
| Quarantine | Exclusion of unsupported/disputed records through review status |
| Robustness set | Separate challenge cases probing ambiguity, unsupported inputs, and manipulation |
| rsLoRA | Rank-stabilized low-rank adapter training |
| Shadow mode | Inspecting predictions while the offline policy withholds decisions |
| Shortcut | Source/template signal that predicts a label without the intended evidence |
| Taxonomy | Versioned class definitions, modes, domains, exclusions, and precedence |
| Temperature | Positive scaling parameter fitted to class-completion scores |
| `test_unseen` | Conditional holdout of scenario families, variants, or sources under the split rules |
| Threshold | Minimum calibrated confidence required for a supported policy decision |
