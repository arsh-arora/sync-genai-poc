# Synchrony Demo Rules & Fixtures Repository

**Purpose:** Provide a complete, rules-ready scaffold and rich dummy content to power a
Synchrony-style multi-agent demo (OfferPilot, TrustShield, Dispute, Collections, Contracts,
DevCopilot, CareCredit, Narrator, ImageGen) across **Consumer** and **Partner** personas.

- This kit is **rules-agnostic** by default. Flip flags in `flags.json` to opt into real rules later.
- All legal/financial texts are **non-binding demo content**.

## Structure
- `flags.json` — feature flags (`demo_no_rules`, etc.).
- `rules/` — declarative rule sets (promos, disclosures, risk, disputes, contracts lexicons, carecredit, narrator metrics, imagegen templates, collections, routing).
- `fixtures/` — demo data for merchants, products, users, transactions, estimates, providers, contracts, marketing templates, dev docs, narrator metrics.
- `golden_tests/` — sample inputs & expected substrings for end-to-end checks.
- `VERSION.txt` — repo version/date.

## Personas
- **Consumer**: offer, dispute, collections, contracts, carecredit (+ narrator-lite). TrustShield runs globally.
- **Partner**: devcopilot, narrator (full), imagegen, contracts (+ offer for promo design). TrustShield runs globally.

## Use
1. Load flags & rules into your app at boot.
2. Use fixtures for offline demo mode to avoid flaky networks.
3. Run golden tests to assert deterministic copy and disclosures.
