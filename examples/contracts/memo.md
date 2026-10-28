# Contract review memo: Tristar Express Lines, Inc.

Agreement: `07-tristar-express.md`

> DRAFT FOR LAWYER REVIEW. Prepared by the contract review agent for Marcus Hale, General Counsel. Nothing has been sent to the carrier. Fictional demo data.

## Summary

The Tristar Express Lines agreement has two deviations from Ridgeline's playbook. First, the carrier's indemnity obligation to Ridgeline is removed, so Ridgeline would bear the risk of carrier negligence while still indemnifying the carrier for Ridgeline's own negligence, willful misconduct, or breach. Second, exclusive venue is set in Davidson County rather than Shelby County, Tennessee. No items were unverifiable by the system, so no additional checks are flagged. Each proposed redline below requires your approval before anything goes to the carrier.

**2 deviation(s)** (1 high severity), **0 item(s) the lawyer must check**, 11 rule(s) compliant.

## Playbook checklist

| Rule | Severity | Status | Contract says | Section |
|---|---|---|---|---|
| Indemnity | high | DEVIATION | ridgeline_indemnifies_carrier | 7.2, 16.4 |
| Venue | medium | DEVIATION | Davidson County, Tennessee | 13.1, 13.2 |
| Cargo liability | high | ok | limit $125,000 | 6.1 |
| Limitation of liability | high | ok | no cap | 9.1 |
| Payment terms | medium | ok | net 30 | 5.2 |
| Non-solicitation | medium | ok | 12 months | 10.1 |
| Governing law | medium | ok | Tennessee | 13.1, 13.2 |
| Termination notice | medium | ok | 30 days | 11.2 |
| Auto-renewal | medium | ok | 30 days to cancel | 11.1 |
| Auto liability insurance | medium | ok | $1,000,000 | 8.1, 8.2 |
| Cargo insurance | medium | ok | $100,000 | 8.1, 8.2 |
| General liability insurance | medium | ok | $1,000,000 | 8.1, 8.2 |
| Certificate holder | low | ok | Ridgeline named | 8.1, 8.2 |

## Deviations and suggested redlines

### 1. Indemnity (high) - §7.2, 16.4

**Playbook:** Carrier indemnifies Ridgeline for carrier negligence. Mutual is acceptable; one-way in the carrier's favor is not.

**Finding (rules engine):** Indemnity does not run from the carrier to Ridgeline (one-way in the carrier's favor).

**Contract text:**

> Notwithstanding Section 7 or anything to the contrary in this Agreement, Carrier shall have no obligation to defend, indemnify or hold harmless Ridgeline or its Customers, and Section 7.1 shall be of no force or effect.
> Ridgeline shall defend, indemnify and hold harmless Carrier from all Losses arising out of Ridgeline's negligence, willful misconduct or breach of this Agreement.

**Plain English:** The contract states that notwithstanding Section 7 or anything else in the Agreement, the carrier has no duty to defend, indemnify, or hold harmless Ridgeline or its Customers, and Section 7.1 is of no effect. Ridgeline, however, must indemnify the carrier for losses arising from Ridgeline's negligence, willful misconduct, or breach. The playbook accepts mutual indemnity but not a one-way indemnity in the carrier's favor, so this clause leaves Ridgeline exposed to carrier-caused losses without recourse.

**Suggested redline** (id `R1`):

> Each party (the 'Indemnifying Party') shall defend, indemnify and hold harmless the other party and its Customers, as applicable, from and against all Losses to the extent arising out of the Indemnifying Party's negligence, willful misconduct, or breach of this Agreement. Section 7.1 shall remain in full force and effect and shall not be limited or overridden by any other provision of this Agreement.

### 2. Venue (medium) - §13.1, 13.2

**Playbook:** Venue: Shelby County, Tennessee.

**Finding (rules engine):** Venue is Davidson County, Tennessee, not Shelby County, Tennessee.

**Contract text:**

> This Agreement is governed by the laws of the State of Tennessee.
> Any litigation arising out of this Agreement shall be brought exclusively in the state or federal courts located in Davidson County, Tennessee.

**Plain English:** The contract is governed by Tennessee law, which the playbook accepts. It requires that any litigation be brought exclusively in state or federal courts in Davidson County, Tennessee. The playbook requires Shelby County, Tennessee, where Ridgeline is based in Memphis, so this clause would move disputes to a forum Ridgeline did not select and may increase travel and litigation cost.

**Suggested redline** (id `R2`):

> This Agreement is governed by the laws of the State of Tennessee. Any litigation arising out of this Agreement shall be brought exclusively in the state or federal courts located in Shelby County, Tennessee, and each party consents to the personal jurisdiction of such courts.
