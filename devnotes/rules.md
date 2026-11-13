# Devnotes rules

## Purpose

- `improvements/`: changes that made it into main and that we stand behind.
- `deadends/`: ideas that were implemented or tested and then dropped. Say why, so they are not retried blindly.
- `open-questions/`: problems or observations without a conclusion yet. A note moves to `improvements/` or `deadends/` once it is settled.
- These are not working logs. A devnote must stand alone: copy in the key numbers.

## Format

- File name: kebab-case topic, e.g. `flux-form-tracers.md`.
- Header block, always in this order:
  - `# Title`: the feature or idea, not the outcome.
  - **Type:** improvement | dead end | open question
  - **Status:** where it stands in main today (default / opt-in / default off / removed), with a date.
  - **Commits:** short hashes or PR numbers, each with its role (feature, merge, default flip).
  - **Code:** types and functions with their files. No line numbers, because they rot.
  - **Docs:** the documentation page, if one exists (optional).
- Links between notes: the file name within the same folder (`ice-conduction.md`), `folder/file.md` across folders.
- Paths: repo-relative for files in Laddie.jl (`src/frazil.jl`). For other public projects, use a GitHub link, e.g. [laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin) for the pan-Antarctic and FRIS runs. Never link private repositories (e.g. the roadmaps), and never use absolute paths on a machine (`/p/...`, `~/...`).
- Length is not limited. Be concise per item, not short overall.
- Sections:
  - Improvement: Summary, Evidence, Parts that were needed (optional), Caveats.
  - Dead end: Idea, Why it is a dead end, What would reopen it.
  - Open question: Question, What we know, Candidate explanations, Next test.

## Content

- Be concise and quantitative. Prefer a table of numbers to a paragraph.
- Every number names its run by its setup (domain, resolution, physics, forcing, the option that differs, day). Never give job IDs: they mean nothing outside the cluster.
- State the comparison baseline explicitly (v2, observations, the previous default).
- A summary is at most 4 sentences. One sentence per line.
- Refer to code and docs (`docs/src/…`) instead of explaining them again.
- Write a dead end as a conclusion ("replaces basal refreezing"), not as a narrative of attempts.
