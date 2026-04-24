# Business Model & Break-Even

Unit economics and path to break-even for the AI Math Tutor, sized for
the Rwandan P1–P3 market. Defended live in the Live Defense.

## Cost base — one cooperative deployment

Each cooperative: **50 tablets shared across 150 children** (3 children
per tablet, as designed in the tablet-sharing model).

| Line item                         | Annual    | Derivation                         |
|-----------------------------------|-----------|------------------------------------|
| Tablets (amortised)               | $1,333    | 50 × $80 ÷ 3-year useful life      |
| Teacher / facilitator training    | $1,500    | 3 villages × $500                  |
| Maintenance + power-bank swaps    |   $500    | $10/tablet/year                    |
| Content updates + curriculum refresh | $500   | 0.1 FTE ÷ 20 deployments           |
| **Total opex per cooperative**    | **$3,833** |                                   |
| **Cost per child per year**       | **$25.50** | $3,833 ÷ 150                      |

No data-plan cost (fully offline by design). No per-diagnosis
inference cost (all on-device; see [footprint_report.md](footprint_report.md)).

## Revenue models & break-even

### A. REB / MINEDUC per-seat licence — **primary path**

- Licence: **$30 per child per year**. Comparable to existing Rwandan
  ed-tech suppliers (Eneza, M-Shule, Bridge International) in the same
  price band.
- Revenue per cooperative: $30 × 150 = **$4,500 / year**.
- Gross margin per cooperative: $4,500 − $3,833 = **$667 / year** (≈ 15%).
- Organisation-level break-even: 8 FTEs × ~$40k = $320k fixed cost ÷
  $667 per-deployment margin = **≈ 480 cooperatives** ≈ **72,000 children**
  ≈ **6 % of Rwanda's P1–P3 enrolment (~1.2 M kids)**.
- Realistic timeline: 24–36 months with government channel.

### B. Cooperative-pay subscription — supplementary only

- $2 / child / month = $3,600 / year per cooperative.
- Unit loss of **$233 / year** — not viable standalone.
- Role: a co-pay layer on top of Model A when REB subsidy is partial.

### C. Grant-funded pilot + government scale — **actual year-0–2 path**

| Year | Phase                                    | Funding / ARR                     |
|------|------------------------------------------|-----------------------------------|
| 1    | Grant pilot: 10 cooperatives (1,500 kids) | $50 k (Mastercard / Imbuto)       |
| 2    | REB LOI: 100 schools (15,000 kids)        | $375 k ARR at $25 /child /year    |
| 3    | Scale to 200 schools (30,000 kids)        | $900 k ARR at $30 /child /year    |

Cumulative cash break-even at **month 14–18** after grant start,
assuming the year-2 REB LOI lands on schedule.

## Value-to-cost — the "is it worth it?" defence

- **World Bank** estimates: +1 mastered year of basic numeracy ≈
  **+10 % lifetime earnings**.
- Rural Rwandan adult median earnings ≈ **$500 / year**.
- NPV of one mastered child: 0.10 × $500 × 40 working years × 0.05
  discount ≈ **$850 per child**.
- Cost for two years of P1–P3 support: 2 × $25.50 = **$51 per child**.
- **ROI = ~17× per child**.

## Risks & mitigations

| Risk                                 | Mitigation                                                    |
|--------------------------------------|---------------------------------------------------------------|
| REB pricing pushback ("teachers are free") | We don't replace teachers — we produce per-child knowledge-tracing teachers cannot do. $30 buys 40+ adaptive sessions/year. |
| Tablet theft / breakage              | 3-year amortisation already absorbs 15 % attrition; cooperative-ownership model reduces theft vs individual ownership. |
| "Why not a free open-source tool?"   | Free tools don't ship Kinyarwanda code-switching, tuned child-voice ASR, weekly parent reports, or dyscalculia flags. The $30 is the maintenance commitment, not the software licence. |
| Grant dependency in year 1           | Secondary revenue channel (Model B co-pay) activates if grant lapses; cooperative-level break-even retained. |
| Device battery / power outages       | Power banks included in maintenance budget; `fsync` on every answered item (see `tutor/storage.py`) — worst-case loss is one item. |

## The 60-second oral defence

> Per-cooperative cost is **$3,833 a year**, or **$25 per child per year** —
> no data fees, fully offline. We break even at deployment level with
> a government per-seat licence at **$30 per child per year**, consistent
> with what existing Rwandan ed-tech suppliers charge. At company
> level with an 8-person team, fixed costs are covered at
> **~480 cooperatives** — **72,000 children** or **6 percent of Rwanda's
> P1–P3 cohort**, achievable in 24 to 36 months through the Rwanda
> Education Board channel. Realistic path is a grant-funded year-one
> pilot at ten cooperatives, then an REB LOI in year two taking us to
> 15,000 children and **$375k ARR**, with cash break-even at month 14
> to 18. On the value side, mastering P1 to P3 numeracy is worth
> roughly **$850 in lifetime-earnings NPV per child** against a
> **$51 two-year cost** — a **17× return**.
