# OpenStreetMap Kerala POIs Dataset — Metadata Report

**Source:** OpenStreetMap via Overpass API  
**URL:** https://overpass-api.de  
**License:** ODbL 1.0 (OpenStreetMap contributors)  
**Acquired:** 2026-05-24 (Overpass API query — Kerala bounding box)  
**Status:** COMPLETE — saved to `data/raw/osm_kerala/`

---

## Files

| File | Description | Size |
|------|-------------|------|
| `kerala_pois.csv` | Flattened POI table | ~8 MB |
| `kerala_pois_raw.json` | Raw Overpass API response | ~35 MB |

---

## Coverage

- **Region:** Kerala, India (area query using OSM relation for Kerala)
- **Query types:** Amenities (nodes) + Infrastructure (street lamps, water wells, parks)
- **Total POIs:** 59,606

---

## Schema

```
osm_id: int         — OpenStreetMap node ID (unique)
lat: float          — latitude (WGS84)
lon: float          — longitude (WGS84)
name: str           — primary name (English / transliterated)
name_ml: str        — name in Malayalam script
amenity: str        — OSM amenity tag
highway: str        — OSM highway tag (street_lamp etc.)
leisure: str        — OSM leisure tag
man_made: str       — OSM man_made tag
landuse: str        — OSM landuse tag
addr_city: str      — city from address tags
addr_district: str  — district from address tags
```

---

## Coverage Statistics

| Metric | Value |
|--------|-------|
| Total POIs | 59,606 |
| With name (English) | 41,911 (70.3%) |
| With name in Malayalam script | 2,390 (4.0%) |
| Districts with data | 15 |

---

## Top 25 POI Categories

| Category | Count |
|----------|-------|
| place_of_worship | 8,997 |
| street_lamp | 5,486 |
| bank | 4,436 |
| childcare | 4,378 |
| restaurant | 4,076 |
| water_well | 3,474 |
| school | 2,782 |
| hospital | 2,631 |
| internet_cafe | 2,548 |
| atm | 2,354 |
| library | 1,409 |
| post_office | 1,403 |
| cafe | 1,194 |
| clinic | 1,150 |
| fuel | 969 |
| pharmacy | 932 |
| fast_food | 752 |
| taxi | 661 |
| conference_centre | 498 |
| college | 496 |
| community_centre | 452 |
| parking | 430 |
| marketplace | 429 |
| toilets | 429 |
| dentist | 416 |

---

## District Coverage (POIs with addr:district tag)

| District | Named POIs |
|----------|-----------|
| Thiruvananthapuram | 240 |
| Ernakulam | 236 |
| Malappuram | 199 |
| Kozhikode | 174 |
| Kollam | 152 |
| Alappuzha | 123 |
| Thrissur | 119 |
| Palakkad | 117 |
| Pathanamthitta | 108 |
| Kannur | 105 |
| Idukki | 83 |
| Kottayam | 82 |
| Wayanad | 41 |
| Kasaragod | 37 |

> Note: Most POIs don't have structured addr:district tags; they are geographically distributed across all 14 Kerala districts.

---

## ML Relevance

| Use Case | Applicability |
|----------|---------------|
| Location extraction | HIGH — landmark names for NER training |
| Geocoding augmentation | HIGH — lat/lon for ward inference from complaint text |
| Civic infrastructure mapping | HIGH — hospitals, schools, police stations match our complaint categories |
| Address normalization | MEDIUM — name variants and aliases |

### Civic-Relevant POI Counts

| Civic Category | OSM Type | Count |
|---|---|---|
| Street light complaints | street_lamp | 5,486 |
| Water supply | water_well | 3,474 |
| Healthcare | hospital + clinic | 3,781 |
| Education | school + college | 3,278 |
| Sanitation | toilets + marketplace | 858 |
| Community space | community_centre + park | ~500 |

---

## Preprocessing Notes

1. **Ward inference:** Use lat/lon + BBMP/ULB ward boundary polygons to assign ward to each POI
2. **Alias extraction:** `name` field contains common landmark names for use in `_LANDMARK_ALIASES`
3. **Deduplication:** Filter by `osm_id` (already done — unique node IDs)
4. **Malayalam name enrichment:** Only 4% have `name_ml` — consider adding via transliteration
5. **Augmentation:** Add ward-POI co-occurrence tables for complaint location inference

---

## Next Steps for Training

1. Extract landmark name list → supplement `_LANDMARK_ALIASES` (245 entries) in `analyzer.py`
2. Build NER training corpus: annotate complaint text with OSM POI names as entity examples
3. Ward inference: spatial join of OSM POIs with ward boundary shapefiles
4. Download ward boundary shapefile: https://data.opencity.in (BBMP ward boundaries GeoJSON)
