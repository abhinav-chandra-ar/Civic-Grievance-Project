# BBMP Grievances Dataset — Metadata Report

**Source:** OpenCity India — CKAN Portal  
**URL:** https://data.opencity.in/dataset/bbmp-grievances-data  
**License:** Open Government Data (OGD) — Government of Karnataka  
**Acquired:** 2026-05-24  
**Status:** COMPLETE

---

## Files

| File | Rows | Size |
|------|------|------|
| `bbmp_grievances_2020.csv` | 91,620 | 12.4 MB |
| `bbmp_grievances_2021.csv` | 103,504 | 13.7 MB |
| `bbmp_grievances_2022.csv` | 118,394 | 15.3 MB |
| `bbmp_grievances_2023.csv` | 119,140 | 15.2 MB |
| `bbmp_grievances_2024.csv` | 207,016 | 26.8 MB |
| `bbmp_grievances_2025.csv` | 126,974 | 17.4 MB |
| **TOTAL** | **766,648** | **~101 MB** |

---

## Schema

```
Complaint ID, Category, Sub Category, Grievance Date,
Ward Name, Grievance Status, Staff Remarks, Staff Name
```

**Column details:**
- `Complaint ID` — unique BBMP complaint identifier string
- `Category` — top-level department category (32 unique values)
- `Sub Category` — specific issue type (181 unique values)
- `Grievance Date` — date submitted
- `Ward Name` — BBMP ward (199 unique wards)
- `Grievance Status` — resolution state
- `Staff Remarks` — free-text official response
- `Staff Name` — assigned officer name

---

## Category Distribution (Top 20)

| Category | Count | % |
|----------|-------|---|
| Electrical | 310,128 | 40.5% |
| Solid Waste (Garbage) Related | 195,153 | 25.5% |
| Road Maintenance(Engg) | 111,535 | 14.5% |
| Forest | 34,618 | 4.5% |
| Health Dept | 29,924 | 3.9% |
| veterinary | 25,524 | 3.3% |
| Road Infrastructure | 12,817 | 1.7% |
| Others | 8,043 | 1.0% |
| Storm Water Drain(SWD) | 6,621 | 0.9% |
| Revenue Department | 5,158 | 0.7% |
| E khata / Khata services | 4,631 | 0.6% |
| Town Planning | 4,622 | 0.6% |
| Parks and Play grounds | 3,857 | 0.5% |
| Advertisement | 2,816 | 0.4% |
| Sanitation | 2,533 | 0.3% |
| Lakes | 2,299 | 0.3% |
| CORONA COVID19 | 1,980 | 0.3% |
| Water Crisis | 963 | 0.1% |
| Markets | 545 | 0.1% |
| Optical Fiber Cables (OFC) | 515 | 0.1% |

**Total unique categories: 32 | Total unique sub-categories: 181**

---

## Top Sub-Categories

| Sub-Category | Count |
|---|---|
| Street Light Not Working | 293,580 |
| Garbage vehicle not arrived | 73,108 |
| Garbage dump | 66,680 |
| Potholes | 32,655 |
| Road side drains | 24,344 |
| Sweeping not done | 20,775 |
| obstructions Branches / Trees. | 17,011 |
| Removal of dead/fallen trees | 15,604 |
| Debris Removal / Construction Material | 14,810 |
| Road Infrastructure | 12,817 |

---

## Status Distribution

| Status | Count |
|--------|-------|
| Closed | 701,878 |
| Registered | 20,404 |
| Rejected | 16,803 |
| Non Relevant | 16,050 |
| Resolved | 6,953 |
| ReOpen | 2,699 |
| In Progress | 1,255 |
| Long Term Solution | 532 |

---

## Coverage

- **Geography:** Bengaluru (BBMP jurisdiction) — 199 wards
- **Time span:** 2020 – 2025 (6 years)
- **Language:** Category/sub-category labels in English; Staff Remarks may contain Kannada

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Complaint classification | HIGH — 32 top-level categories, 181 sub-categories map directly to our department taxonomy |
| Priority inference | MEDIUM — status field + date can be used to derive SLA urgency patterns |
| Category-to-dept routing | HIGH — category names align with our `_CATEGORY_TO_DEPT` mapping |
| Label alignment | MEDIUM — BBMP categories need manual mapping to our 10 internal categories |

### Category Mapping to Our System

| BBMP Category | Our Internal Category |
|---|---|
| Electrical | street_light, electrical_hazard |
| Solid Waste (Garbage) Related | waste_management, solid_waste |
| Road Maintenance(Engg) | road_damage |
| Storm Water Drain(SWD) | drainage, sewage_issue |
| Parks and Play grounds | tree_fall |
| Water Crisis | water_supply |

---

## Next Steps for Training

1. Map BBMP categories to our 10 internal categories (manual crosswalk needed)
2. Use `Sub Category` as fine-grained signal for complaint classification
3. Filter out `Others`, `CORONA COVID19` as noisy labels
4. Use `Staff Remarks` as augmentation training text (free-text civic complaints)
5. Balance classes — Electrical is heavily over-represented (40%)
