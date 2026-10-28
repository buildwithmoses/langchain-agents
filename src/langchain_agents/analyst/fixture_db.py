"""Deterministic Ridgeline TMS fixture: a SQLite copy of the tables ops managers ask about.

    uv run python fixtures/analyst/generate.py        # (re)build fixtures/analyst/ridgeline.db

Six full months (2026-04-01 .. 2026-09-30), ~2,400 loads/month, 600 shippers,
1,800 carriers. Same seed -> byte-identical facts every time.

Planted patterns (what a good analyst should find):
1. Lane Memphis, TN -> Chicago, IL (FTL dry van): ~14% margin April-August,
   NEGATIVE margin in September 2026 (carrier costs jumped, shipper contract rate did not).
2. Carrier "Bluegrass Haulers LLC": ~45% of deliveries late vs ~10% network-wide.
   Late loads are always 61+ minutes past the appointment, on-time loads arrive at
   or before it, so any sensible "late" definition gives the same ranking.
3. Detention at "Harbor Foods DC - Atlanta" (shipper Harbor Foods Inc.) spikes in
   August and September 2026 (~55% of deliveries vs ~6% elsewhere).
Also: ~3% of loads are Cancelled (rate and cost zeroed), a few late-September loads are
still In Transit, and claims cluster on reefer loads.

PII columns (contact and driver identity) are fictional: 555-01xx phones, @example.com emails.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260401
START = datetime(2026, 4, 1)
MONTHS = [(2026, m) for m in range(4, 10)]  # April..September 2026
LOADS_PER_MONTH = 2400
N_SHIPPERS = 600
N_CARRIERS = 1800

PLANTED_LANE = ("Memphis", "TN", "Chicago", "IL")
PLANTED_LATE_CARRIER = "Bluegrass Haulers LLC"
PLANTED_DETENTION_SHIPPER = "Harbor Foods Inc."
PLANTED_DETENTION_FACILITY = "Harbor Foods DC - Atlanta"

# Columns that identify a person. The SQL guard and the output scan read this map.
PII_COLUMNS: dict[str, set[str]] = {
    "carriers": {"contact_name", "contact_phone", "contact_email"},
    "shippers": {"billing_contact", "billing_email"},
    "loads": {"driver_name", "driver_phone"},
}

SCHEMA = """
CREATE TABLE shippers (
  shipper_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  industry TEXT NOT NULL,
  city TEXT NOT NULL,
  state TEXT NOT NULL,
  payment_terms_days INTEGER NOT NULL,
  onboarded_at TEXT NOT NULL,
  billing_contact TEXT NOT NULL,
  billing_email TEXT NOT NULL
);
CREATE TABLE facilities (
  facility_id INTEGER PRIMARY KEY,
  shipper_id INTEGER NOT NULL REFERENCES shippers(shipper_id),
  name TEXT NOT NULL,
  facility_type TEXT NOT NULL,          -- 'DC', 'Plant', 'Warehouse'
  city TEXT NOT NULL,
  state TEXT NOT NULL
);
CREATE TABLE carriers (
  carrier_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  mc_number TEXT NOT NULL,
  dot_number TEXT NOT NULL,
  safety_rating TEXT NOT NULL,          -- 'Satisfactory', 'Unrated', 'Conditional'
  auto_liability_limit INTEGER NOT NULL,
  cargo_insurance_limit INTEGER NOT NULL,
  home_state TEXT NOT NULL,
  fleet_size INTEGER NOT NULL,
  onboarded_at TEXT NOT NULL,
  contact_name TEXT NOT NULL,
  contact_phone TEXT NOT NULL,
  contact_email TEXT NOT NULL
);
CREATE TABLE loads (
  load_id INTEGER PRIMARY KEY,
  shipper_id INTEGER NOT NULL REFERENCES shippers(shipper_id),
  carrier_id INTEGER REFERENCES carriers(carrier_id),
  pickup_facility_id INTEGER REFERENCES facilities(facility_id),
  delivery_facility_id INTEGER REFERENCES facilities(facility_id),
  origin_city TEXT NOT NULL,
  origin_state TEXT NOT NULL,
  dest_city TEXT NOT NULL,
  dest_state TEXT NOT NULL,
  mode TEXT NOT NULL,                   -- 'FTL', 'LTL'
  equipment TEXT NOT NULL,              -- 'Dry Van', 'Reefer', 'Flatbed'
  miles INTEGER NOT NULL,
  weight_lbs INTEGER NOT NULL,
  pickup_appt TEXT NOT NULL,            -- 'YYYY-MM-DD HH:MM'
  pickup_actual TEXT,
  delivery_appt TEXT NOT NULL,
  delivery_actual TEXT,
  shipper_rate REAL NOT NULL,           -- USD billed to shipper (linehaul)
  carrier_cost REAL NOT NULL,           -- USD paid to carrier (linehaul)
  status TEXT NOT NULL,                 -- 'Delivered', 'In Transit', 'Cancelled'
  driver_name TEXT,
  driver_phone TEXT
);
CREATE TABLE check_calls (
  check_call_id INTEGER PRIMARY KEY,
  load_id INTEGER NOT NULL REFERENCES loads(load_id),
  called_at TEXT NOT NULL,
  city TEXT NOT NULL,
  state TEXT NOT NULL,
  status TEXT NOT NULL                  -- 'At Pickup', 'In Transit', 'Delayed', 'At Delivery'
);
CREATE TABLE accessorials (
  accessorial_id INTEGER PRIMARY KEY,
  load_id INTEGER NOT NULL REFERENCES loads(load_id),
  type TEXT NOT NULL,                   -- 'Detention', 'Layover', 'Lumper'
  stop TEXT NOT NULL,                   -- 'pickup', 'delivery'
  hours REAL,                           -- detention: total hours on site (2 free)
  amount REAL NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE claims (
  claim_id INTEGER PRIMARY KEY,
  load_id INTEGER NOT NULL REFERENCES loads(load_id),
  filed_at TEXT NOT NULL,
  claim_type TEXT NOT NULL,             -- 'Damage', 'Shortage', 'Loss'
  amount_claimed REAL NOT NULL,
  amount_paid REAL NOT NULL,
  status TEXT NOT NULL                  -- 'Open', 'Paid', 'Denied'
);
CREATE INDEX ix_loads_shipper ON loads(shipper_id);
CREATE INDEX ix_loads_carrier ON loads(carrier_id);
CREATE INDEX ix_loads_pickup ON loads(pickup_appt);
CREATE INDEX ix_loads_lane ON loads(origin_city, dest_city);
CREATE INDEX ix_cc_load ON check_calls(load_id);
CREATE INDEX ix_acc_load ON accessorials(load_id);
CREATE INDEX ix_claims_load ON claims(load_id);
CREATE INDEX ix_fac_shipper ON facilities(shipper_id);
"""

CITIES = [
    ("Memphis", "TN", 35.15, -90.05), ("Nashville", "TN", 36.16, -86.78), ("Chicago", "IL", 41.88, -87.63),
    ("Atlanta", "GA", 33.75, -84.39), ("Dallas", "TX", 32.78, -96.80), ("Houston", "TX", 29.76, -95.37),
    ("St. Louis", "MO", 38.63, -90.20), ("Kansas City", "MO", 39.10, -94.58), ("Indianapolis", "IN", 39.77, -86.16),
    ("Louisville", "KY", 38.25, -85.76), ("Little Rock", "AR", 34.75, -92.29), ("Jackson", "MS", 32.30, -90.18),
    ("Birmingham", "AL", 33.52, -86.80), ("Charlotte", "NC", 35.23, -80.84), ("Columbus", "OH", 39.96, -83.00),
    ("Detroit", "MI", 42.33, -83.05), ("Jacksonville", "FL", 30.33, -81.66), ("Orlando", "FL", 28.54, -81.38),
    ("New Orleans", "LA", 29.95, -90.07), ("Oklahoma City", "OK", 35.47, -97.52), ("Cincinnati", "OH", 39.10, -84.51),
    ("Savannah", "GA", 32.08, -81.09), ("Laredo", "TX", 27.51, -99.51), ("Denver", "CO", 39.74, -104.99),
    ("Phoenix", "AZ", 33.45, -112.07), ("Harrisburg", "PA", 40.27, -76.88), ("Allentown", "PA", 40.60, -75.47),
    ("Joliet", "IL", 41.53, -88.08), ("Fort Worth", "TX", 32.76, -97.33), ("Greenville", "SC", 34.85, -82.40),
]
STATES = sorted({c[1] for c in CITIES} | {"WI", "MN", "IA", "NE", "VA", "WV", "NJ", "NY", "CA", "WA"})
INDUSTRIES = ["Food & Beverage", "Retail", "Manufacturing", "Automotive", "Building Materials", "Paper & Packaging",
              "Consumer Goods", "Chemicals (non-hazmat)", "Agriculture", "Electronics"]
FIRST = ["James", "Maria", "Robert", "Linda", "Michael", "Patricia", "David", "Jennifer", "William", "Elizabeth",
         "Carlos", "Aisha", "Kevin", "Tamika", "Jose", "Megan", "Darnell", "Rosa", "Brian", "Keisha", "Luis", "Amber",
         "Andre", "Hannah", "Marcus", "Yolanda", "Tyler", "Nicole", "Victor", "Crystal"]
LAST = ["Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Wilson",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Moore", "Allen", "Young",
        "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Campbell", "Mitchell"]
CARRIER_A = ["Blue Ridge", "Delta", "Iron Horse", "Prairie", "Gulf Coast", "Big River", "Summit", "Redline", "Eagle",
             "Lone Star", "Heartland", "Ozark", "Magnolia", "Cumberland", "Pioneer", "Granite", "Crossroads",
             "Midsouth", "Silver Spur", "Northstar", "Bayou", "Timberline", "Liberty", "Copperhead", "Riverbend"]
CARRIER_B = ["Transport", "Freight", "Logistics", "Trucking", "Carriers", "Express", "Hauling", "Lines", "Transit"]
SHIPPER_A = ["Acme", "Southern", "Valley", "Pinnacle", "Brightway", "Cedar", "Oakmont", "Riverside", "Keystone",
             "Highland", "Summit", "Harvest", "Atlas", "Evergreen", "Frontier", "Golden", "Lakeside", "Meridian",
             "Northfield", "Pacific", "Quality", "Redwood", "Sterling", "Tristar", "United", "Vista", "Westbrook"]
SHIPPER_B = ["Foods", "Supply", "Industries", "Manufacturing", "Brands", "Products", "Packaging", "Distribution",
             "Materials", "Goods", "Mills", "Farms"]
AREA_CODES = ["901", "615", "312", "404", "214", "713", "314", "816", "317", "502", "501", "601", "205", "704"]


def _ts(d: datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M")


def _miles(a: tuple, b: tuple) -> int:
    # Rough road miles: great-circle-ish * 1.2 road factor.
    import math

    lat1, lon1, lat2, lon2 = map(math.radians, (a[2], a[3], b[2], b[3]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return max(40, int(3959 * 2 * math.asin(math.sqrt(h)) * 1.2))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _phone(rng: random.Random) -> str:
    return f"({rng.choice(AREA_CODES)}) 555-01{rng.randrange(100):02d}"


def _email(name: str, domain: str, rng: random.Random) -> str:
    first, last = name.lower().split()
    return f"{first[0]}{last}{rng.randrange(10, 99)}@{domain}.example.com"


def build_db(path: Path) -> Path:
    rng = random.Random(SEED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    city_by_name = {(c[0], c[1]): c for c in CITIES}

    # --- shippers + facilities -------------------------------------------------
    shippers, facilities = [], []
    used = set()
    fac_id = 0
    shipper_facs: dict[int, list[int]] = {}
    fac_city: dict[int, tuple] = {}
    for sid in range(1, N_SHIPPERS + 1):
        if sid == 7:
            name = PLANTED_DETENTION_SHIPPER
        else:
            while True:
                name = f"{rng.choice(SHIPPER_A)} {rng.choice(SHIPPER_B)} {rng.choice(['Inc.', 'LLC', 'Co.', 'Corp.'])}"
                if name not in used and name != PLANTED_DETENTION_SHIPPER:
                    break
        used.add(name)
        home = rng.choice(CITIES)
        contact = _person(rng)
        shippers.append((sid, name, "Food & Beverage" if sid == 7 else rng.choice(INDUSTRIES), home[0], home[1],
                         rng.choice([30, 30, 30, 45, 60]), _ts(datetime(2014, 1, 1) + timedelta(days=rng.randrange(4300))),
                         contact, _email(contact, "shipper" + str(sid), rng)))
        n_fac = 3 if sid == 7 else rng.choice([1, 1, 2, 2, 3])
        shipper_facs[sid] = []
        for k in range(n_fac):
            fac_id += 1
            if sid == 7:
                c = [city_by_name[("Atlanta", "GA")], city_by_name[("Memphis", "TN")], city_by_name[("Dallas", "TX")]][k]
                fname = [PLANTED_DETENTION_FACILITY, "Harbor Foods Plant - Memphis", "Harbor Foods DC - Dallas"][k]
                ftype = ["DC", "Plant", "DC"][k]
            else:
                c = home if k == 0 else rng.choice(CITIES)
                ftype = rng.choice(["DC", "Plant", "Warehouse"])
                fname = f"{name.rsplit(' ', 1)[0]} {ftype} - {c[0]}"
            facilities.append((fac_id, sid, fname, ftype, c[0], c[1]))
            shipper_facs[sid].append(fac_id)
            fac_city[fac_id] = c
    con.executemany("INSERT INTO shippers VALUES (?,?,?,?,?,?,?,?,?)", shippers)
    con.executemany("INSERT INTO facilities VALUES (?,?,?,?,?,?)", facilities)
    harbor_dc = shipper_facs[7][0]

    # --- carriers ----------------------------------------------------------------
    carriers, cused = [], set()
    for cid in range(1, N_CARRIERS + 1):
        if cid == 42:
            name = PLANTED_LATE_CARRIER
        else:
            while True:
                name = f"{rng.choice(CARRIER_A)} {rng.choice(CARRIER_B)} {rng.choice(['LLC', 'Inc.', 'Co.'])}"
                if rng.random() < 0.5:
                    name = name.replace(" LLC", f" {rng.randrange(2, 99)} LLC")
                if name not in cused and name != PLANTED_LATE_CARRIER:
                    break
        cused.add(name)
        contact = _person(rng)
        rating = "Satisfactory" if rng.random() < 0.78 else ("Unrated" if rng.random() < 0.8 else "Conditional")
        carriers.append((cid, name, f"MC{100000 + cid * 37 % 900000:06d}", f"{2000000 + cid * 131:07d}", rating,
                         rng.choice([1_000_000, 1_000_000, 1_000_000, 2_000_000, 5_000_000]),
                         rng.choice([100_000, 100_000, 250_000, 250_000, 500_000]),
                         rng.choice(STATES), max(1, int(rng.paretovariate(1.2) * 3)),
                         _ts(datetime(2012, 1, 1) + timedelta(days=rng.randrange(5000))),
                         contact, _phone(rng), _email(contact, "carrier" + str(cid), rng)))
    con.executemany("INSERT INTO carriers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", carriers)
    # 40 core carriers haul ~half the freight; carrier 42 (Bluegrass) is one of them.
    core = [42] + rng.sample([c for c in range(1, N_CARRIERS + 1) if c != 42], 39)

    # Shipper weights: a few big accounts, long tail. Harbor Foods is a top-10 account.
    shipper_w = [rng.paretovariate(1.1) for _ in range(N_SHIPPERS)]
    shipper_w[6] = sorted(shipper_w)[-6]

    lanes_hot = [("Memphis", "TN", "Chicago", "IL"), ("Memphis", "TN", "Atlanta", "GA"), ("Memphis", "TN", "Dallas", "TX"),
                 ("Chicago", "IL", "Memphis", "TN"), ("Dallas", "TX", "Houston", "TX"), ("Atlanta", "GA", "Charlotte", "NC"),
                 ("Memphis", "TN", "St. Louis", "MO"), ("Nashville", "TN", "Atlanta", "GA"), ("Laredo", "TX", "Dallas", "TX"),
                 ("Indianapolis", "IN", "Columbus", "OH")]

    loads, check_calls, accessorials, claims = [], [], [], []
    load_id = cc_id = acc_id = claim_id = 0
    for year, month in MONTHS:
        first = datetime(year, month, 1)
        nxt = datetime(year + (month == 12), month % 12 + 1, 1)
        days = (nxt - first).days
        for _ in range(LOADS_PER_MONTH):
            load_id += 1
            sid = rng.choices(range(1, N_SHIPPERS + 1), weights=shipper_w)[0]
            planted_lane = False
            r = rng.random()
            if r < 0.025 and sid != 7:
                o_city, o_state, d_city, d_state = PLANTED_LANE
                planted_lane = True
                equipment, mode = "Dry Van", "FTL"
            else:
                if r < 0.25:
                    o_city, o_state, d_city, d_state = rng.choice(lanes_hot[1:])
                else:
                    a, b = rng.sample(CITIES, 2)
                    o_city, o_state, d_city, d_state = a[0], a[1], b[0], b[1]
                mode = "LTL" if rng.random() < 0.18 else "FTL"
                equipment = "Dry Van" if mode == "LTL" else rng.choices(["Dry Van", "Reefer", "Flatbed"], [0.6, 0.25, 0.15])[0]
            if (o_city, d_city) == ("Memphis", "Chicago") and not planted_lane:
                # keep the planted lane clean: only planted rows use it
                d_city, d_state = "Joliet", "IL"
            facs = shipper_facs[sid]
            if sid == 7:
                # Harbor Foods ships from its Memphis plant into its DCs, mostly Atlanta.
                o_city, o_state = "Memphis", "TN"
                pick_fac = facs[1]
                deliv_fac = harbor_dc if rng.random() < 0.7 else facs[2]
                d_city, d_state = fac_city[deliv_fac][0], fac_city[deliv_fac][1]
                mode, equipment = "FTL", rng.choice(["Reefer", "Reefer", "Dry Van"])
            else:
                pick_fac = facs[0] if rng.random() < 0.6 else None
                deliv_fac = None
            miles = _miles(city_by_name[(o_city, o_state)], city_by_name[(d_city, d_state)])
            weight = rng.randint(1500, 12000) if mode == "LTL" else rng.randint(18000, 44000)

            pickup = first + timedelta(days=rng.randrange(days), hours=rng.choice([6, 7, 8, 9, 10, 13, 14, 15]))
            transit_h = max(4, int(miles / 45) + rng.randint(1, 6)) + (24 if mode == "LTL" else 0)
            delivery_appt = pickup + timedelta(hours=transit_h)

            rpm = {"Dry Van": 2.35, "Reefer": 2.80, "Flatbed": 2.95}[equipment]
            if mode == "LTL":
                rate = round(180 + weight * 0.045 + miles * 0.35 * rng.uniform(0.85, 1.15), 2)
            else:
                rate = round((250 + miles * rpm) * rng.uniform(0.9, 1.12), 2)
            margin = rng.uniform(0.10, 0.19)
            if planted_lane:
                rate = round(1650 * rng.uniform(0.98, 1.02), 2)  # contract rate, flat all six months
                margin = rng.uniform(0.12, 0.16) if month < 9 else -rng.uniform(0.04, 0.09)
            cost = round(rate * (1 - margin), 2)

            carrier = rng.choice(core) if rng.random() < 0.5 else rng.randint(1, N_CARRIERS)
            if planted_lane and carrier == 42:
                carrier = core[1]

            status = "Delivered"
            if rng.random() < 0.03:
                status = "Cancelled"
            elif delivery_appt > datetime(2026, 9, 30, 12):
                status = "In Transit"

            if status == "Cancelled":
                loads.append((load_id, sid, None, pick_fac, deliv_fac, o_city, o_state, d_city, d_state, mode, equipment,
                              miles, weight, _ts(pickup), None, _ts(delivery_appt), None, 0.0, 0.0, status, None, None))
                continue

            pickup_actual = pickup + timedelta(minutes=rng.randint(-30, 45))
            late_p = 0.45 if carrier == 42 else 0.10
            if rng.random() < late_p:
                delivery_actual = delivery_appt + timedelta(minutes=rng.randint(61, 480))
            else:
                delivery_actual = delivery_appt - timedelta(minutes=rng.randint(0, 90))
            if status == "In Transit":
                delivery_actual_s = None
            else:
                delivery_actual_s = _ts(delivery_actual)
            driver = _person(rng)
            loads.append((load_id, sid, carrier, pick_fac, deliv_fac, o_city, o_state, d_city, d_state, mode, equipment,
                          miles, weight, _ts(pickup), _ts(pickup_actual), _ts(delivery_appt), delivery_actual_s,
                          rate, cost, status, driver, _phone(rng)))

            # check calls: every ~4h in transit, capped
            n_calls = min(6, max(1, transit_h // 8 + rng.randint(0, 1)))
            for k in range(n_calls):
                cc_id += 1
                t = pickup_actual + timedelta(hours=k * transit_h / max(1, n_calls))
                if k == 0:
                    cc_status = "At Pickup"
                    c, st = o_city, o_state
                elif k == n_calls - 1 and status == "Delivered" and n_calls > 1:
                    cc_status = "At Delivery"
                    c, st = d_city, d_state
                else:
                    cc_status = "Delayed" if (delivery_actual > delivery_appt and rng.random() < 0.5) else "In Transit"
                    c, st = (o_city, o_state) if rng.random() < 0.5 else (d_city, d_state)
                check_calls.append((cc_id, load_id, _ts(t), c, st, cc_status))

            if status != "Delivered":
                continue
            # detention: delivery stop
            det_p = 0.06
            if deliv_fac == harbor_dc:
                det_p = 0.55 if month >= 8 else 0.07
            for stop, p in (("pickup", 0.04), ("delivery", det_p)):
                if rng.random() < p:
                    hours = round(2 + rng.choice([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 9]), 2)
                    billable = min(600.0, 75.0 * (int((hours - 2) * 4) / 4))
                    acc_id += 1
                    at = delivery_actual if stop == "delivery" else pickup_actual
                    accessorials.append((acc_id, load_id, "Detention", stop, hours, round(billable, 2),
                                         _ts(at + timedelta(hours=hours))))
            if rng.random() < 0.01:
                acc_id += 1
                accessorials.append((acc_id, load_id, "Layover", "delivery", None, 350.0,
                                     _ts(delivery_actual + timedelta(days=1))))
            if equipment in ("Reefer", "Dry Van") and rng.random() < 0.05:
                acc_id += 1
                accessorials.append((acc_id, load_id, "Lumper", "delivery", None, round(rng.uniform(75, 450), 2),
                                     _ts(delivery_actual + timedelta(hours=1))))
            claim_p = 0.012 if equipment == "Reefer" else 0.004
            if rng.random() < claim_p:
                claim_id += 1
                amt = round(rng.uniform(400, 18000), 2)
                cst = rng.choices(["Open", "Paid", "Denied"], [0.3, 0.5, 0.2])[0]
                paid = round(amt * rng.uniform(0.6, 1.0), 2) if cst == "Paid" else 0.0
                claims.append((claim_id, load_id, _ts(delivery_actual + timedelta(days=rng.randint(1, 40))),
                               rng.choice(["Damage", "Damage", "Shortage", "Loss"]), amt, paid, cst))

    con.executemany("INSERT INTO loads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", loads)
    con.executemany("INSERT INTO check_calls VALUES (?,?,?,?,?,?)", check_calls)
    con.executemany("INSERT INTO accessorials VALUES (?,?,?,?,?,?,?)", accessorials)
    con.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?)", claims)
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(path)
    return path


def ensure_db(path: Path) -> Path:
    """Build the fixture on first use (it is deterministic, so this is safe)."""
    path = Path(path)
    if not path.exists():
        build_db(path)
    return path
