# Data

The original course dataset is **not included** in this portfolio repository because the source materials do not specify redistribution rights.

To reproduce the analysis, place the original course file here:

```text
data/raw/euphoria_dataset.csv
```

Expected schema:

```text
referral_friends
water_sources
shelters
fauna_friendly
island_size
creation_time
region
happiness_metric
features
happiness_index
loyalty_score
total_refunds_requested
trade_goods
x_coordinate
avg_time_in_euphoria
y_coordinate
island_id
entry_fee
nearest_city
```

The supplied file contains 99,492 observations. Some records are stored as a single quoted CSV field; `src/data_cleaning.py` handles both formats.

If you own the rights to redistribute the dataset, you can add it to the repository yourself. The `.gitignore` intentionally excludes files under `data/raw/` by default.
