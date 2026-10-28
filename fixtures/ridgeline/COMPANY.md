# Ridgeline Freight (fictional customer, demo scenario)

Every agent in this repo is built for the same pretend customer. Use these facts; don't invent conflicting ones. All names, numbers, phone numbers (555-01xx) and emails (@example.com) are fictional.

## Who they are
- Third-party freight broker in Memphis, TN. 140 employees. Founded 2014.
- They don't own trucks. They match **shippers** (customers with freight) to **carriers** (trucking companies) and earn the margin.
- Modes: full truckload (dry van, reefer, flatbed) and LTL (less than truckload). About 2,400 loads a month across 600 active shippers and 1,800 approved carriers.
- Systems: a TMS (transportation management system) backed by Postgres; a help desk; a shared drive of carrier and shipper contracts.

## Policies (the support agent and contract reviewer rely on these)
- **Quotes** are valid for 24 hours. Spot quotes over $8,000 need a pricing manager's sign-off.
- **Tracking**: check calls every 4 hours while a load is in transit; shippers get an ETA update if a load slips more than 2 hours.
- **Detention**: 2 hours free at pickup and at delivery, then $75/hour, billed in 15-minute increments, capped at $600 per stop. Requires in/out times on the signed BOL (bill of lading).
- **Layover**: $350 per day when a carrier is held overnight through no fault of their own.
- **Claims** (cargo loss or damage): must be filed within 9 months of delivery with photos, the BOL noting exceptions, and an invoice for the goods. Ridgeline files against the carrier's cargo insurance; it does not pay claims itself. Acknowledgement within 2 business days; resolution typically 30 to 60 days.
- **Payment terms**: shippers net 30. Carriers net 30, or quick pay in 2 days for a 3% fee.
- **Credits and refunds**: only the billing team can issue them. Support never promises one.
- **Hazmat**: Ridgeline does not broker hazmat loads.
- **Carrier requirements**: active USDOT and MC authority, at least $1,000,000 auto liability and $100,000 cargo insurance, a satisfactory or unrated safety rating, and 6 months of operating history.

## Contract playbook (what legal wants in every carrier agreement)
- Carrier indemnifies Ridgeline for carrier negligence. Mutual indemnity is acceptable; one-way indemnity in the carrier's favor is not.
- Cargo liability at full value up to at least $100,000 per load. Carmack-style liability; no "released rates" that cap liability below $100,000 without legal sign-off.
- Payment terms no shorter than net 30 for Ridgeline to pay. Quick pay is optional.
- No back-solicitation of Ridgeline's shippers by the carrier for 12 months after the last load.
- Governing law: Tennessee. Venue: Shelby County.
- Either party may terminate on 30 days' written notice. Anything longer than 60 days needs legal sign-off.
- Auto-renewal is acceptable if notice to cancel is 30 days or less.
- Insurance: auto liability at least $1,000,000, cargo at least $100,000, general liability at least $1,000,000, Ridgeline named as certificate holder.
- Limitation of liability caps that apply to the carrier's own negligence are not acceptable.

## People
- Dana Reyes, VP Operations (sponsor for the support and analyst POCs)
- Marcus Hale, General Counsel (contract review)
- Priya Nair, Head of Data (SQL analyst)
- Support desk: support@ridgeline.example.com, 555-0142
