# Estonian Cadastre — Field & Term Glossary

Working glossary for the Tallinn / Estonia CivicMapper integration. Source dataset:
**Maa-amet (Estonian Land Board) national cadastre** — `Eesti_KATASTER_GPKG.zip`
(`https://s3.pilw.io/rp-kemit-kataster/ANDMED/20260101_Eesti_KATASTER_GPKG.zip`),
one layer `Eesti`, **775,185** parcels nationwide, **EPSG:3301** (L-EST97 / Estonian
national grid — must reproject to EPSG:4326).

## Cadastre columns (GPKG layer `Eesti`)

| Field | Estonian | English | Notes / use in CivicMapper |
|---|---|---|---|
| `tunnus` | katastritunnus | Cadastral unit code | Parcel ID, e.g. `78401:001:0018`. First block `784xx` = Tallinn cadastral districts. |
| `hkood` | haldusüksuse kood | Admin-unit code (EHAK) | Municipality code (Tallinn = 0784). |
| `mk_nimi` | maakonna nimi | County name | Tallinn → `Harju maakond`. |
| `ov_nimi` | omavalitsuse nimi | Municipality name | **Filter key**: `ov_nimi = 'Tallinn'` → 38,616 parcels. |
| `ay_nimi` | asustusüksuse nimi | Settlement / district name | For Tallinn = **linnaosa** (city district); 8 of them → good region grouping. |
| `l_aadress` | lähiaadress | Address | Street + number, e.g. `Oblika tee 43`. |
| `registr` | registreeritud | Registered date | Parcel registration date. |
| `muudet` | muudetud | Modified date | Last modification. |
| `siht1/2/3` | sihtotstarve 1/2/3 | Intended-use purpose(s) | Land-use category. A parcel can have up to 3 mixed uses. `siht1` is primary. |
| `so_prts1/2/3` | sihtotstarbe protsent | Purpose percentage | % of the parcel in each `sihtN` use (sums to 100). |
| `pindala` | pindala | Area | **m²** (integer). Basis for per-area value. |
| `haritav` | haritav maa | Arable land | m² breakdown (rural). |
| `rohumaa` | rohumaa | Grassland | m² breakdown. |
| `mets` | mets | Forest | m² breakdown. |
| `ouemaa` | õuemaa | Yard / homestead land | m² breakdown. |
| `muumaa` | muu maa | Other land | m² breakdown. |
| `kinnistu` | kinnistu | Land-register (property) no. | Links to Kinnistusraamat (land register). |
| `muutpohjus` | muutmise põhjus | Reason for change | Administrative. |
| `omvorm` | omandivorm | **Ownership form** | `Eraomand` (private), `Munitsipaalomand` (municipal), `Riigiomand` (state), `Avalik-õiguslik omand` (public-law), `Segaomand` (mixed). Proxy for "exempt/public". |
| `maks_hind` | maksustamishind | **Taxable / assessed value (€)** | **The land value.** Post-2022 mass-revaluation basis for the land tax (maamaks). Median Tallinn ≈ €186k/parcel, ≈ €168/m². Infrastructure/no-purpose land floored at €10/m². **No separate building value exists — Estonia does not tax buildings.** |
| `marked` | märked | Notes / markings | Free text. |
| `ads_oid`, `adob_id` | ADS objekti id | Address-system (ADS) IDs | Join keys to the national address system. |
| `oiguslik_alus` | õiguslik alus | Legal basis | Basis of registration. |
| `eksport` | eksport | Export date | Dataset export stamp. |

## Sihtotstarve (land-use purpose) codes seen in Tallinn

| `siht1` value | Estonian meaning | English | Count (Tallinn) | Proposed category |
|---|---|---|---|---|
| `ELAMUMAA` | elamumaa | Residential land | 26,403 | Residential |
| `TRANSPORDIMAA` | transpordimaa | Transport land (roads/streets) | 3,806 | Transport / ROW |
| `TOOTMISMAA` | tootmismaa | Production / industrial land | 2,718 | Industrial |
| `ARIMAA` | ärimaa | Commercial / business land | 2,420 | Commercial |
| `ULDKASUTATAV_MAA` | üldkasutatav maa | Public / common-use land (parks) | 1,594 | Open Space / Public |
| `UHISKONDLIKE_EHITISTE_MAA` | ühiskondlike ehitiste maa | Land for public/civic buildings | 826 | Civic / Institutional |
| `SIHTOTSTARBETA_MAA` | sihtotstarbeta maa | Land with no designated purpose | 545 | Undesignated |
| `MAATULUNDUSMAA` | maatulundusmaa | Agricultural / yield land (farm+forest) | 205 | Agricultural |
| `RIIGIKAITSEMAA` | riigikaitsemaa | National-defence land | 53 | Defense (exempt) |
| `KAITSEALUNE_MAA` | kaitsealune maa | Protected land | 18 | Protected (exempt) |
| `MAETOOSTUSMAA` | mäetööstusmaa | Mining / extraction land | 12 | Mineral |
| `VEEKOGUDE_MAA` | veekogude maa | Water-body land | 7 | Water |
| `JAATMEHOIDLA_MAA` | jäätmehoidla maa | Waste-storage land | 6 | Utility |
| `SOTSIAALMAA` | sotsiaalmaa | Social land | 3 | Social |

## Administrative geography

- **Tallinn** = a single `omavalitsus` (municipality), `ov_nimi = 'Tallinn'`, code 0784, in
  **Harju maakond** (county).
- Tallinn's 8 **linnaosad** (city districts, in `ay_nimi`): Nõmme, Pirita, Haabersti,
  Kesklinn, Põhja-Tallinn, Kristiine, Lasnamäe, Mustamäe. (Below these are ~84 **asumid**
  / neighbourhoods, not in the cadastre — available from Tallinn open data / OSM.)

## Key facts that shape the ETL

1. **Land value, not building value.** `maks_hind` is assessed *land* value only. CivicMapper's
   primary metric (`REALLANDVA_per_sqft`, land value per sqft) maps directly; `REALIMPROV`
   (improvements) has no source. The improvement/land-ratio "Underdeveloped" heuristic can't be
   computed from the cadastre alone — it needs a building-footprint source (Overture/EHR).
2. **One parcel per land unit.** Apartment buildings are a single `katastriüksus`; individual
   flats are *korteriomandid* in the land register, not separate cadastral polygons. So there is
   **no US-style per-unit condo stacking** to collapse.
3. **Filter by attribute, not geometry.** `ov_nimi = 'Tallinn'` is authoritative — no spatial
   clip against a boundary shapefile needed.
4. **CRS EPSG:3301 → 4326** reprojection is required before export.
5. **Infrastructure floor.** Roads (`TRANSPORDIMAA`) and undesignated land get a flat €10/m²
   assessed value; left in, they flatten the low end of the value ramp.
