# Clinical Safety and Human Oversight

## Current status

REHAB AI v0.1.0 is research and portfolio software. It has not undergone prospective or retrospective clinical validation, regulatory review, usability validation with clinicians, fairness evaluation on patient subgroups, or medical-device quality-system verification.

## Safety architecture

The software intentionally separates prediction from decision. A predicted risk is not directly emitted as a treatment instruction. Candidate interventions pass through scenario simulation and explicit feasibility constraints before ranking. Every result is marked `clinician_review_required`.

## Baseline constraints implemented

- maximum therapy-session duration;
- lower intensity under high predicted fall risk;
- lower intensity under high fatigue;
- lower intensity under high pain;
- minimum balance work when configured fall-risk threshold is exceeded;
- session-frequency restriction under severe fatigue;
- explicit review flag for assistive-device consideration.

These constraints are engineering examples and are not a substitute for guideline-derived protocol logic or clinician judgment.

## Prohibited claims

Until dataset-backed clinical validation exists, project documentation must not claim diagnostic accuracy, improved patient outcomes, reduced falls, optimized treatment efficacy in real patients, or autonomous treatment capability.

## Required next clinical work

Before any real-world use, define intended use and target population, involve qualified rehabilitation clinicians, derive protocol constraints from authoritative clinical sources, validate sensor measurement error, validate model calibration and subgroup behavior, conduct scenario and human-factors testing, implement access control/auditing, and establish a formal clinical risk-management process.
