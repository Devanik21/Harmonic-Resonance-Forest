# Harmonic Resonance Fields (HRF) - Agent Guidelines

Greetings, fellow researcher or automated agent. This repository documents the Harmonic Resonance Forest (HRF), a physics-informed machine learning framework. To maintain the scientific integrity and technical excellence of this project, please adhere to the following guidelines.

# AGENTS.md

## Repository Purpose

Harmonic-Resonance-Forest is a research-oriented EEG and resonance analysis framework with experimental AI-assisted maintenance workflows.

---

## Contributor Policy

For community programs such as GSSoC:

* Human contributors must implement and submit their own pull requests.
* AI agents may assist with suggestions, review, documentation drafting, benchmarking ideas, or planning.
* Autonomous AI-generated pull requests for contributor-assigned issues should be avoided.

---

## Jules Behavior Rules

When working on GSSoC-related issues:

* Do not automatically create pull requests.
* Prefer generating implementation plans, review comments, architecture suggestions, or draft documentation only.
* Wait for explicit maintainer instruction before opening a PR.

---

## Labels

* `gssoc-2026` → Human contributor priority
* `jules` → AI collaboration allowed
* `internal-automation` → Jules autonomous execution allowed (very rare and only if explicit only)

---

## Documentation Standards

* Preserve scientific terminology and research accuracy.
* Avoid oversimplifying resonance or EEG processing concepts.
* Maintain repository aesthetic and technical tone.

---

## Workflow Preference

Preferred workflow:

Issue → Discussion → Human PR → AI Review Assistance → Maintainer Review

Avoid:

Issue → Autonomous AI PR without contributor involvement

---

## Additional Notes

AI agents may still:

* review code
* suggest refactors
* generate diagrams
* improve architecture notes
* propose benchmarks
* assist maintainers internally

without directly replacing contributor participation.


## Maintenance Principles

- **Minimalist Intervention:** Prefer the smallest possible high-impact edit over large rewrites.
- **Scientific Trustworthiness:** Ensure all mathematical claims and benchmarks are strictly verified.
- **Preserve Intent:** Respect the author's framing of classification as a physical resonance problem.
- **Environment Hygiene:** Build artifacts, environment-specific files (like `__pycache__`), and binary bytecode must never be committed. Ensure `.gitignore` is maintained.
- **Safety First:** Security vulnerabilities should be reported via private channels.

---
*Inspired by the engineering culture of Google DeepMind.*
