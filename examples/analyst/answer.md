# How many loads were cancelled in September 2026?

Assuming "in September" means loads with a pickup appointment in September 2026, 74 loads were cancelled. I filtered on `pickup_appt` because cancelled loads may never have an actual pickup time.

**SQL** (LIMIT 200 enforced)
```sql
SELECT COUNT(*) AS cancelled_loads FROM loads WHERE status = 'Cancelled' AND pickup_appt >= '2026-09-01' AND pickup_appt < '2026-10-01'
LIMIT 200
```

**Result** - 1 row(s)

| cancelled_loads |
|---|
| 74 |

---

# And how many in August, compared with September?

August had 84 cancelled loads and September had 74, so September had fewer cancellations. Both counts use the pickup appointment date, the same basis as the September figure above.

**SQL** (LIMIT 200 enforced)
```sql
SELECT strftime('%Y-%m', pickup_appt) AS month, COUNT(*) AS cancelled_loads FROM loads WHERE status = 'Cancelled' AND pickup_appt >= '2026-08-01' AND pickup_appt < '2026-10-01' GROUP BY month ORDER BY month
LIMIT 200
```

**Result** - 2 row(s)

| month | cancelled_loads |
|---|---|
| 2026-08 | 84 |
| 2026-09 | 74 |
