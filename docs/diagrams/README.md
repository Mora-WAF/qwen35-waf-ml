# Diagram sources and verification

[Learning index](../README.md)

`pipeline.html` is an interactive architecture overview of the implemented offline pipeline. Open it directly in a browser. It needs no application server; an optional external font may fall back to the system font offline.

`pipeline.architecture.json` is its editable Archify specification. The numbered path covers source evidence, preparation/review, dataset construction, the gate, packaging, Colab training/scoring, calibration, and policy. It is a conceptual component map, not a deployed service topology or a claim that full training has completed.

The Markdown chapters contain Mermaid topology for classification, dataset partitions, the training lifecycle, and policy branches. The three SVG teaching visuals in `../assets/` need no JavaScript or external resources. They include accessible titles and descriptions. Dataset counts are a dated snapshot; the loss mask and minimal pair are explicitly schematic.

## Regeneration

If Archify is installed, run its CLI from the skill directory, using absolute paths for the project files:

```bash
node bin/archify.mjs validate architecture <project>/docs/diagrams/pipeline.architecture.json --quality showcase --json
node bin/archify.mjs deliver architecture <project>/docs/diagrams/pipeline.architecture.json <project>/docs/diagrams/pipeline.html --quality showcase --json
node bin/archify.mjs visual-check <project>/docs/diagrams/pipeline.html --json
```

If the environment prevents child-process creation, the renderer and artifact checker can be invoked directly as separate shell commands. That produces a checked artifact but does not provide the CLI's atomic-delivery receipt:

```bash
node renderers/architecture/render-architecture.mjs <project>/docs/diagrams/pipeline.architecture.json <project>/docs/diagrams/pipeline.html
node scripts/check-render-output.mjs <project>/docs/diagrams/pipeline.html
```

See [validation.json](validation.json) for the actual method, hashes, and checks for this delivered version. Browser inspection and structural checks are separate evidence; an unavailable check is recorded as unavailable.
