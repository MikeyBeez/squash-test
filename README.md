# Squash Test: Context Window Extension Experiment

Testing whether pretrained language models can handle context windows larger than their training size without retraining.

## Hypothesis
Models trained on N tokens can handle >N tokens with graceful degradation rather than catastrophic failure.

## Experiment
Test GPT-2 (trained on 1024 tokens) with extended context sizes up to 2048+ tokens.

## Significance
If successful, this enables immediate context scaling without model retraining.
