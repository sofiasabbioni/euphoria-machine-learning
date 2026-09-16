
import csv

import numpy as np
import pandas as pd

from src.data_cleaning import _parse_row, engineer_features


def test_parse_embedded_csv_row():
    original = [
        "3.0",
        "2.0",
        "2.0",
        "Cats,Dogs",
        "892.0",
        "1568754491.0",
        "TX",
        "Monthly",
        "AC,Pool,Storage",
        "1014.0",
        "2.0",
        "1.0",
        "USD",
        "32.8239",
        "39.84",
        "-96.681",
        "5508811414.0",
        "No",
        "Dallas",
    ]
    embedded = next(csv.reader([",".join([
        '3.0','2.0','2.0','"Cats,Dogs"','892.0','1568754491.0',
        'TX','Monthly','"AC,Pool,Storage"','1014.0','2.0','1.0',
        'USD','32.8239','39.84','-96.681','5508811414.0','No','Dallas'
    ])]))
    # When a complete record is presented as one field, a second parse should
    # reconstruct the original 19 columns.
    wrapped = [",".join([
        '3.0','2.0','2.0','"Cats,Dogs"','892.0','1568754491.0',
        'TX','Monthly','"AC,Pool,Storage"','1014.0','2.0','1.0',
        'USD','32.8239','39.84','-96.681','5508811414.0','No','Dallas'
    ])]
    assert _parse_row(wrapped, 19) == original


def test_engineered_features():
    df = pd.DataFrame(
        {
            "creation_time": [1568754491.0],
            "features": ["AC,Pool,Storage"],
            "fauna_friendly": ["Cats,Dogs"],
        }
    )
    out = engineer_features(df)
    assert int(out.loc[0, "creation_year"]) == 2019
    assert int(out.loc[0, "amenity_count"]) == 3
    assert int(out.loc[0, "cats_allowed"]) == 1
    assert int(out.loc[0, "dogs_allowed"]) == 1
