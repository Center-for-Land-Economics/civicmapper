# Attribution & Data Sources

Civic Mapper visualizes publicly available parcel geometry and property-assessment
data. Every city in the app is built from that city's (or its county's / country's)
own public data source; this file lists each source, the dataset used, and its
license or terms as far as we have been able to determine them.

Accuracy notes:

- Where a source states an explicit license (CC0, CC BY 4.0, ODbL, …) it is named.
- Most US county assessor / appraisal-district data is **public record whose portal
  does not state a formal license**. For those rows the table says
  *"public record — license/terms not formally stated; see source portal"* rather
  than inventing one. Check the linked portal for its current terms before reusing.
- If you spot an error or a missing/incorrect attribution, please open an issue —
  corrections are very welcome.

---

## Basemaps & derived layers

These apply app-wide, on top of the per-city parcel data:

| Layer | Source | License |
|---|---|---|
| Basemap tiles ("OpenStreetMap" style) | © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Basemap tiles ("OpenFreeMap" style) | © [OpenFreeMap](https://openfreemap.org/) · © OpenStreetMap contributors | Data: [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Basemap tiles ("OpenTopoMap" style) | © OpenStreetMap contributors, SRTM · map style © [OpenTopoMap](https://opentopomap.org/) | Style: [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); data: ODbL |
| Surface parking lot footprints | © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (parking/fuel tags) | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) — attribution required on display |
| Building footprints (surface-vs-structure parking classification) | © [Overture Maps Foundation](https://overturemaps.org/) (Buildings theme) | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) — see [docs.overturemaps.org/attribution](https://docs.overturemaps.org/attribution/) |

The in-app parking layer carries the combined attribution string
`© OpenStreetMap contributors · Building footprints © Overture Maps Foundation (ODbL)`.
See `docs/parking-surface-vs-structure.md` for the ODbL share-alike analysis of the
derived parking dataset.

---

## Parcel & assessment data by city

### United States

| City | Jurisdiction / Source agency | Dataset | License / Terms | Link |
|---|---|---|---|---|
| Albuquerque, NM | Bernalillo County Assessor (+ City of Albuquerque open data for context/boundary) | ASROnline Public Map parcel service | Public record — license/terms not formally stated; see source portal | [assessormap.bernco.gov](https://assessormap.bernco.gov/server/rest/services/GIS/ASROnline_Public_Map/MapServer) · [data.cabq.gov](https://data.cabq.gov/) |
| Austin, TX | Travis Central Appraisal District (TCAD) + Travis County | TCAD appraisal-roll export (PROP.TXT, manual download) joined to Travis County taxmaps parcel geometry | Public record — no published terms restrict access or reuse (checked 2026-07); provided as-is. TCAD's export ZIPs are free anonymous downloads; in practice Cloudflare blocks scripted fetches, so pull the ZIP manually (robots.txt asks for a 10s crawl delay) | [traviscad.org/publicinformation](https://traviscad.org/publicinformation) · [taxmaps.traviscountytx.gov](https://taxmaps.traviscountytx.gov/arcgis/rest/services/Parcels/MapServer/0) |
| Baltimore, MD | City of Baltimore GIS (attributes from Maryland SDAT assessments) | CityView "Realproperty_OB" real-property parcel layer | Public record — license/terms not formally stated; see source portal | [geodata.baltimorecity.gov](https://geodata.baltimorecity.gov/egis/rest/services/CityView/Realproperty_OB/FeatureServer/0) |
| Bellingham, WA | Whatcom County Assessor via Whatcom County GIS (neighborhood regions from the City of Bellingham AGOL "Bellingham_Neighborhoods" layer) | "Public Tax Parcels" layer (geometry + market/appraised land & improvement values); *reconstructed* — the parcel ETL predates this repo, but this county layer matches the shipped land/improvement fields; the county's bulk Property Data Downloads are the alternative extract | Public record — county data-use disclaimer applies, and WA law (RCW 42.56.070) bars commercial use of lists of individuals; see source portal | [gis.whatcomcounty.us](https://gis.whatcomcounty.us/arcgis/rest/services/EnterprisePublishing/WhatcomCo_Property/MapServer/1) · [Property Data Downloads](https://www.whatcomcounty.us/3869/Property-Data-Downloads) |
| Bryan / College Station, TX | Brazos County / Brazos Central Appraisal District (hosted ArcGIS layer) | "Brazos County, Texas Parcels" (geometry + values + class) + "Brazos County City Limits" | Public record — license/terms not formally stated; see source portal | [Parcels](https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/Brazos_County_Parcels/FeatureServer/0) · [City limits](https://services1.arcgis.com/qr14biwnHA6Vis6l/arcgis/rest/services/Brazos_County_City_Limits/FeatureServer/0) |
| Charlottesville, VA | City of Charlottesville Open Data GIS | OpenData_1 / OpenData_2 parcel, assessment-history, and detail layers | Public record — license/terms not formally stated; see source portal | [gisweb.charlottesville.org](https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_1/MapServer) |
| Chicago, IL | Cook County (Assessor + County GIS) via the county open-data (Socrata) portal | "ccgisdata – Parcel 2021" (77tz-riq7, geometry) + "Assessor – Assessed Values" (uzyt-m557); assessed values converted to market via the classification-ordinance level of assessment | Public record — see Cook County Data Portal terms of use | [datacatalog.cookcountyil.gov](https://datacatalog.cookcountyil.gov/) |
| Cincinnati, OH | Hamilton County Auditor via CAGIS Open Data | Hamilton County Parcel Polygons (CAGIS_Open_Data) | Public record — license/terms not formally stated; see source portal | [CAGIS Open Data](https://services.arcgis.com/JyZag7oO4NteHGiq/ArcGIS/rest/services) |
| Cleveland, OH | Cuyahoga County | MyPLACE parcel layer (Parcels_WMA_GJOIN_WGS84) | Public record — license/terms not formally stated; see source portal | [gis.cuyahogacounty.us](https://gis.cuyahogacounty.us/server/rest/services/MyPLACE/Parcels_WMA_GJOIN_WGS84/MapServer/2) |
| Columbus, OH | City of Columbus GIS (aggregating the central-Ohio county auditors: Franklin, Delaware, Fairfield used here; FCAO value schema + Ohio DTE class codes) | "Central Ohio Parcels" (KeyLayers/3) + City of Columbus Corporate Boundary | Public record — license/terms not formally stated; see source portal | [maps.columbus.gov](https://maps.columbus.gov/arcgis/rest/services/CityServices/KeyLayers/MapServer/3) |
| Dallas, TX | City of Dallas GIS + Dallas Central Appraisal District (DCAD) | "DallasTaxParcels" FeatureServer (geometry/class) joined to the DCAD appraisal-roll export ACCOUNT_APPRL_YEAR (manual download) | Public record — no published terms restrict access or reuse (checked 2026-07; dallascad.org has no terms-of-use page); provided as-is without warranty. Bulk ZIPs are free anonymous downloads; DCAD's robots.txt disallows crawling per-account pages, so use the bulk files | [gis.dallascityhall.com](https://gis.dallascityhall.com/arcgis/rest/services/Basemap/DallasTaxParcels/FeatureServer/0) · [dallascad.org](https://www.dallascad.org/DataProducts.aspx) |
| Denver, CO | City & County of Denver — Denver Open Data Catalog | ODC_PROP_PARCELS_A (parcels with assessment attributes) | Public record — see Denver Open Data Catalog terms | [Denver open data parcels](https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245) |
| Detroit, MI | City of Detroit Office of the Assessor — Detroit Open Data (data.detroitmi.gov) | "Parcels (Current)" (geometry/class) + "Tentative Assessment Roll 2026" (values), joined on parcel_id | Public record — see Detroit open-data portal terms | [data.detroitmi.gov](https://data.detroitmi.gov/) |
| Fort Collins, CO | Larimer County (GIS + Assessor public data center) | Parcels MapServer (geometry) + assessor public CSVs (account / value-detail / improvement) | Public record — license/terms not formally stated; see source portal | [maps1.larimer.org](https://maps1.larimer.org/arcgis/rest/services/MapServices/Parcels/MapServer/3) · [larimer.gov/assessor/publicdata](https://www.larimer.gov/assessor/publicdata) |
| Houston, TX (also the dev-only "Harris County" build) | Harris Central Appraisal District (HCAD) | HCAD Parcels service (geometry + values + state class) + HCAD_Cities municipal boundaries; bulk alternative: Parcels.zip + Real_acct_owner.zip from download.hcad.org | Public record — license/terms not formally stated; see HCAD data downloads | [gis.hctx.net](https://www.gis.hctx.net/arcgis/rest/services/HCAD/Parcels/MapServer/0) · [download.hcad.org](https://download.hcad.org/) |
| Lynchburg, VA | City of Lynchburg GIS Office / City Assessor (independent city — layer is inherently city-only) | Open Data Portal "Parcels" (OpenData/ODPDynamic MapServer layer 41 — geometry + current assessor land/improvement/total values + property class/description + acreage) | Public record. The publisher states: "neither a legally recorded map nor a survey"; see the City's [GIS Data Sharing and Distribution Policy](http://www.lynchburgva.gov/gis-data-sharing-and-distribution-policy) | [Open Data Portal item](https://data-cityoflynchburg.opendata.arcgis.com/datasets/56b924d2dd8a404a8830fc8b0612febe) · [MapServer layer 41](https://mapviewer.lynchburgva.gov/ArcGIS/rest/services/OpenData/ODPDynamic/MapServer/41) |
| Morgantown, WV | West Virginia Integrated Assessment System (WV IAS) + Monongalia County GIS | WV IAS property-record API (assessed values, Monongalia County) + Monongalia_AGOL parcel geometry | Public record — license/terms not formally stated; WV IAS API is key-gated — see wvias.io | [wvias.io](https://wvias.io/) |
| New York City, NY (and the "NYC (IBX)" corridor subset) | NYC Department of City Planning | MapPLUTO (all five boroughs; IBX build is a walkshed subset of the same layer) | NYC Open Data — see the MapPLUTO data-use terms on the DCP/NYC Open Data page | [MapPLUTO](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page) |
| Newport News, VA | City of Newport News GIS (independent city — layer is inherently city-only) | Operational/Parcel MapServer (geometry + current assessor values + class) | Public record — license/terms not formally stated; see source portal | [maps.nnva.gov](https://maps.nnva.gov/gis/rest/services/Operational/Parcel/MapServer/0) |
| Olympia, WA | Thurston County GeoData + City of Olympia | Countywide Parcels layer (geometry + assessor values), clipped to the City of Olympia municipal boundary | Public record — license/terms not formally stated; see source portal | [Thurston County parcels](https://tconline.co.thurston.wa.us/server/rest/services/Common_Layers/Parcels/FeatureServer/4) |
| Portland, OR | Multnomah County (DART — Division of Assessment, Recording & Taxation) | Taxlots_Orion_Public (taxlots with valuation fields) + LevyCode city boundaries | Public record — license/terms not formally stated; see source portal | [www3.multco.us](https://www3.multco.us/gisagspublic/rest/services/DART/Taxlots_Orion_Public/MapServer/0) |
| Pueblo, CO | Pueblo County GIS | PuebloCounty_Parcels + municipal/county boundaries + zoning layers | Public record — license/terms not formally stated; see source portal | [maps.co.pueblo.co.us](https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_Parcels/MapServer/1) |
| Rochester, NY | City of Rochester (assessor data via the DataROC open-data portal) | "City of Rochester Tax Parcel Records: Open Data" (Tax_Parcels_Open_Data — citywide geometry + CURRENT_LAND_VALUE / CURRENT_TOTAL_VALUE etc.); *reconstructed* — the ETL predates this repo, but this layer matches the shipped "current_*" value fields; tax/exemption attributes may additionally have drawn on the NYS Open NY assessment roll | Public record — "made available by the City of Rochester, NY for public use"; see DataROC terms | [data.cityofrochester.gov](https://data.cityofrochester.gov/) · [Tax parcels layer](https://maps.cityofrochester.gov/server/rest/services/Open_Data/Tax_Parcels_Open_Data/FeatureServer/0) |
| Rockville, MD | Maryland iMAP (statewide services; assessment attributes from Maryland SDAT) | MD_ParcelBoundaries (PlanningCadastre) + MD_PoliticalBoundaries "Municipal Boundaries – Detailed" | Public record — see Maryland iMAP terms | [mdgeodata.md.gov](https://mdgeodata.md.gov/imap/rest/services/PlanningCadastre/MD_ParcelBoundaries/MapServer/0) |
| San Antonio, TX | Bexar Appraisal District (BCAD) data via hosted ArcGIS layers + City of San Antonio GIS | "Bexar_parcels_all" (geometry + values + class) + CoSAGIS "BCAD_Parcels" (exemption flags) + "COSABoundary" city limits | Public record — license/terms not formally stated; see source portals | [Bexar parcels](https://services2.arcgis.com/82iS1Pc7dgs3LFZv/arcgis/rest/services/Bexar_parcels_all/FeatureServer/0) |
| Seattle, WA | King County GIS / King County Assessor | "Parcels for King County with Address, Property and Ownership Information" (PARCEL_ADDRESS_PUB_AREA), clipped to the City of Seattle boundary (KingCo_AdministrativeAreas) | Public record — see King County GIS data terms | [King County parcels](https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0) |
| South Bend, IN | St. Joseph County, IN GIS (SJCGIS — the county's hosted ArcGIS Online org, which also serves the South Bend open-data hub) | "Parcel_Civic" (assessment attributes: land/improvement values, tax district, property type, tax-info links) + "parcel_boundaries" (geometry), filtered to PROP_CITY = "SOUTH BEND" — confirmed from the recovered ETL notebook | Public record — license/terms not formally stated; see source portal | [SJCGIS hub](https://sjcgis-stjocogis.hub.arcgis.com/) · [Parcel_Civic](https://services.arcgis.com/OjftlhRHkAABcyiF/ArcGIS/rest/services/Parcel_Civic/FeatureServer) |
| Spokane, WA | Spokane County (Assessor data via the county's hosted ArcGIS "Parcels" layer; SCOUT property system for detail links), filtered to the "Spokane General" levy tax-code areas | Spokane County Parcels FeatureServer | Public record — license/terms not formally stated; see source portal | [Spokane County parcels](https://services1.arcgis.com/ozNll27nt9ZtPWOn/ArcGIS/rest/services/) |
| St. Paul, MN | Ramsey County Open Data | Ramsey County OpenData parcel FeatureServer (assessment attributes; property-tax lookup links) | Public record — see Ramsey County open-data terms | [maps.co.ramsey.mn.us](https://maps.co.ramsey.mn.us/arcgis/rest/services/OpenData/OpenData/FeatureServer) |
| Syracuse, NY | City of Syracuse Open Data (geometry: Onondaga County Planning; assessment attributes: City of Syracuse Department of Assessment) | "Syracuse Parcel Map" (quarterly citywide parcel layer with land_av / total_av assessed values); *reconstructed* — the ETL predates this repo, but this layer matches the shipped fields; tax/exemption attributes may additionally have drawn on the NYS Open NY assessment roll ("Property Assessment Data from Local Assessment Rolls") | Public record — license/terms not formally stated on the dataset pages; see source portal | [data.syr.gov](https://data.syr.gov/) · [Open NY assessment rolls](https://data.ny.gov/Government-Finance/Property-Assessment-Data-from-Local-Assessment-Rol/7vem-aaz7) |
| Tulsa, OK | Tulsa County Assessor, published by INCOG (Indian Nations Council of Governments) | Parcels_TulsaCo FeatureServer (geometry + assessor values + classification), filtered to SiteCity='TULSA' | Public record — license/terms not formally stated; see source portal | [map11.incog.org](https://map11.incog.org/arcgis11wa/rest/services/Parcels_TulsaCo/FeatureServer/0) |
| Vancouver, WA | Clark County GIS / Clark County Assessor | ClarkView_Public "Taxlots" layer (geometry + market land/building values + use class + jurisdiction), JurisDesc='Vancouver' | Public record — license/terms not formally stated; see source portal | [gis.clark.wa.gov](https://gis.clark.wa.gov/arcgisfedpw/rest/services/ClarkView_Public/Taxlots/MapServer/0) |
| Washington, DC | Office of the Chief Technology Officer / OCFO — **Open Data DC** | "Common Ownership Lots" (Owner Polygons, layer 40) + ITSPE tax-roll extract + CONDORELATE condo-unit table | **CC0 / public domain (stated by Open Data DC)** | [opendata.dc.gov](https://opendata.dc.gov/) |
| DC · Maryland · Virginia ("DMV" — **dev-only, not publicly deployed**) | DC OCTO (Owner Polygons), Maryland iMAP/SDAT (Montgomery + Prince George's), Fairfax County GIS, Arlington County GIS, City of Fairfax GIS | One stitched multi-jurisdiction build; each jurisdiction from its own public ArcGIS service | Mixed — DC portion CC0; MD/VA portions public record, see each portal | [Open Data DC](https://opendata.dc.gov/) · [Maryland iMAP](https://imap.maryland.gov/) · [Fairfax County GIS](https://www.fairfaxcounty.gov/maps/) · [Arlington GIS](https://gisdata-arlgis.opendata.arcgis.com/) |

### Europe

| City | Jurisdiction / Source agency | Dataset | License / Terms | Link |
|---|---|---|---|---|
| Tallinn, Estonia | Maa-amet (Estonian Land Board) national cadastre | Bulk cadastre GPKG (Eesti KATASTER), filtered to Tallinn; `maks_hind` = assessed **land** value only (Estonia has no building assessment). District boundaries from the official EHAK settlement-unit polygons. | Maa-amet open data — published as an open bulk download; attribution to Maa-amet; see the Maa-amet geoportal for exact terms | [Maa-amet geoportal](https://geoportaal.maaamet.ee/) |
| Copenhagen, Denmark | Geometry: DAWA / Dataforsyningen (Danish public basic data). Values: Vurderingsstyrelsen via Datafordeler VUR | `jordstykker` cadastral parcels (kommunekode 0101) joined by BFE number to Ejendomsvurdering (VUR) grundværdi/ejendomsværdi | Geometry: open Danish basic data ("grunddata"), no key required — see dataforsyningen.dk terms. Values: free but **registration-gated** (Datafordeler API key required) — see datafordeler.dk terms | [api.dataforsyningen.dk](https://api.dataforsyningen.dk/) · [datafordeler.dk](https://datafordeler.dk/) |
| Stockholm, Sweden — **in progress, not shipped** (geometry-only ETL exists; no city entry in the app) | Geometry: Lantmäteriet "Fastighetsindelning Nedladdning, vektor". Values: not yet sourced (taxeringsvärde access is agreement-gated via Skatteverket / Lantmäteriet) | Property-division polygons (Stockholms kommun), locally downloaded GeoPackage | Geometry: **CC0 (stated)** — free High-Value Data since 2025, but the download itself requires a free opendata.lantmateriet.se account. Values: **blocked/agreement-gated**, not yet used | [opendata.lantmateriet.se](https://opendata.lantmateriet.se/) |

---

## Data redistribution note

The underlying parcel/assessment datasets are **not redistributed in this
repository**. This repo contains ETL code, the visualization frontend, and
configuration; the processed parcel parquets and tile sets are built from the
sources above and served separately (blob storage / CDN).

If you deploy your own instance of Civic Mapper, **you** take on the attribution
and terms-of-use obligations of every source you serve: the OSM/Overture ODbL
attribution (kept in the map's attribution control and the parking layer — do not
remove it), the CC BY attribution for Kartverket data, the stated terms of each
assessor/open-data portal, and any registration or permission requirements
(e.g. the Datafordeler API key for Copenhagen values).
