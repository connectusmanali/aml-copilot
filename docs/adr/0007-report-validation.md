# ADR 0007: Cited STR drafting with a deterministic validator

- Status: accepted
- Date: 2026-10-20

## Context
An STR narrative that states a wrong amount or date is worse than no draft. LLM self-checks
and LLM judges are not reliable enough to be the control.

## Decisions
1. **Every sentence cites evidence ids**; the drafting prompt allows only the case file.
2. **A deterministic validator decides**, not a model: citations exist; policy cannot be the
   only citation; every amount, date and account in a sentence must appear in the cited
   records (or be the sum of the cited transactions, or a figure the rule analyzer computed
   from exactly those records).
3. **One regeneration** with the validator's findings as feedback; if it still fails, the case
   goes to `needs_human` with the problems highlighted. A template drafter (valid by
   construction) is the offline mode and the degraded mode.
4. **Investigator edits are validated too**; only a validating report can be approved as final,
   and filing requires a final report. Edit distance (draft → final) is stored per decision.
5. **Mutation testing** in the eval harness plants wrong amounts, dates, accounts and citations
   in every draft; the gate requires 100% caught.
