# To do

Open items found along the way. Newest first; move items out when done.

## Parking model licence: NVIDIA non-commercial terms (found 2026-11-17)

The pipeline's parking model, `UTEL-UIUC/SegFormer-large-parking`, is tagged Apache 2.0 on Hugging
Face but is a fine-tune of `nvidia/segformer-b5-finetuned-ade-640-640`, whose weights are under the
NVIDIA Source Code License for SegFormer (github.com/NVlabs/SegFormer/blob/master/LICENSE). Section
3.3 limits "the Work and any derivative works thereof" to non-commercial use, defined as "research
or evaluation purposes only". Our own fine-tunes inherit this. Publishing citywide parking maps
from it may fall outside "research or evaluation"; get a legal read before relying on it.
Unencumbered replacements to test in the bake-off: DINOv2 (Apache 2.0), Clay (Apache 2.0),
DINOv3-sat (Meta DINOv3 License: commercial use allowed, attribution and pass-through terms).

## Houston: impossible parcel land areas (found 2026-11-15)

The hosted `parquets-dev/houston-tx-parcels.parquet` sums to ~15,100 km² of parcel land, about
nine times the city (~1,650 km²). Almost all of it is two "Other" accounts whose land area comes
from HCAD's reported roll acreage (`area_source = reported`), not the drawn polygon:

| HCAD account | land_area_acres | land value |
|---|---|---|
| [0432060000718](https://public.hcad.org/records/details.asp?crypt=0432060000718) | 3,368,996 (≈13,600 km²) | $0 |
| [0432060000223](https://public.hcad.org/records/details.asp?crypt=0432060000223) | 48,400 | $0 |

Houston's "Other" category totals 3.45 million acres across 61,731 rows (median 0 acres).

To check:
- What these accounts are (likely right-of-way, utility or a roll placeholder) and whether the roll
  acreage is in square feet or simply wrong; compare with the polygon area.
- Add a sanity rule to the pipeline: prefer the GIS area when the reported acreage exceeds the
  polygon by more than ~10x (or exceeds the city's area), and log the cases. The Texas pipeline
  already has "polygon is a fragment of the roll's land area" logic; this is the opposite case.
- What downstream numbers the bad acreage feeds: land totals (`*-land-totals.json`), acreage-based
  shares and per-acre figures in the app, and any hexagon aggregates using `land_area_acres`.
- Whether other pre-existing cities (San Antonio, Austin) have similar outliers.

Found while estimating citywide surface parking; that estimate excluded Houston's "Other" land.

## Parking labels (NAIP model)

- Save the exempt parcels the Texas pipeline drops (schools, churches, government land) to a side
  file so the label tool, label export and parking-share figures can show exempt land and flag
  parking on it. About 10% of Laredo's labelled parking is on land with no parcel in our data.
- Laredo: finish the 150 random squares, then the 25 targeted squares; retrain after each ~25.
- Learning curve: add a train-on-a-fraction option to `label_finetune.py` to see when more labels
  stop helping.
- Second labeller on the double-labelled squares (agreement sets the accuracy ceiling).
- Plano campaign: test whether a Laredo-trained model transfers.
- Citywide surface-parking estimate per city from the random squares (stratified, model-assisted).
