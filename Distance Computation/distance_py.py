import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_required_columns(df, required_columns, file_label):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{file_label} is missing required columns: {sorted(missing_columns)}"
        )


def haversine_distance_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def add_extreme_outlier_columns(
    df,
    division_col="division",
    school_lat_col="latitude",
    school_lon_col="longitude",
    distance_col="linear_distance_km",
    outlier_percentile_col="distance_outlier_percentile",
    nearby_count_col="nearby_school_count",
    outlier_col="coordinate_outlier_category",
    min_schools_per_division=5,
    extreme_percentile=99,
    nearby_radius_km=5,
):
    """
    Adds outlier columns based on linear_distance_km within each division.

    Main rule:
    - A school is an extreme outlier if its linear_distance_km is in the 99th
      percentile or higher within its division.

    nearby_school_count:
    - Counts other schools in the same division within nearby_radius_km.
    - Default radius is 5 km.
    - This is kept as a diagnostic column only. It does not suppress outlier
      detection.

    coordinate_outlier_category:
    - extreme_outlier
    - normal
    - missing_distance
    - insufficient_division_data
    """

    df[outlier_percentile_col] = pd.NA
    df[nearby_count_col] = pd.NA
    df[outlier_col] = "normal"

    valid_coordinates = (
        df[school_lat_col].notna()
        & df[school_lon_col].notna()
    )

    # Count nearby peer schools within the same division.
    for division, group in df.loc[valid_coordinates].groupby(division_col, dropna=False):
        lats = group[school_lat_col].to_numpy()
        lons = group[school_lon_col].to_numpy()

        for row_index, lat, lon in zip(group.index, lats, lons):
            distances_to_peers = haversine_distance_km(lat, lon, lats, lons)

            # Subtract 1 to exclude the school itself.
            nearby_count = int((distances_to_peers <= nearby_radius_km).sum() - 1)
            df.loc[row_index, nearby_count_col] = nearby_count

    missing_distance = df[distance_col].isna()
    df.loc[missing_distance, outlier_col] = "missing_distance"

    usable_rows = df.loc[~missing_distance]

    for division, group in usable_rows.groupby(division_col, dropna=False):
        if len(group) < min_schools_per_division:
            df.loc[group.index, outlier_col] = "insufficient_division_data"
            continue

        # Percentile rank within the division.
        # Higher distance = more outlier.
        percentiles = (
            group[distance_col]
            .rank(method="max", pct=True)
            .mul(100)
            .round(2)
        )

        df.loc[group.index, outlier_percentile_col] = percentiles

        extreme_candidates = percentiles >= extreme_percentile
        df.loc[group.index[extreme_candidates], outlier_col] = "extreme_outlier"

    return df


def add_school_to_sdo_distance(
    school_csv="ph_schools_202606121415.csv",
    sdo_csv="sdo_classification_202606151013.csv",
    output_csv="ph_schools_with_distance_and_outliers.csv",
    match_key="match_key",
    division_col="division",
    school_lat_col="latitude",
    school_lon_col="longitude",
    sdo_lat_col="latitude",
    sdo_lon_col="longitude",
    distance_col="linear_distance_km",
    outlier_percentile_col="distance_outlier_percentile",
    nearby_count_col="nearby_school_count",
    outlier_col="coordinate_outlier_category",
    nearby_radius_km=5,
):
    """
    Creates a new CSV with:
    - all original columns from ph_schools
    - linear_distance_km
    - distance_outlier_percentile
    - nearby_school_count
    - coordinate_outlier_category

    Outlier detection:
    - First, compute each school's distance to its matched SDO.
    - Then, rank that distance against other schools in the same division.
    - 99+ percentile means extreme distance.
    - Nearby school count is retained for context only and does not suppress
      the outlier tag.
    """

    school_csv = Path(school_csv)
    sdo_csv = Path(sdo_csv)
    output_csv = Path(output_csv)

    school_df = pd.read_csv(school_csv, low_memory=False)
    sdo_df = pd.read_csv(sdo_csv, low_memory=False)

    original_school_columns = list(school_df.columns)

    validate_required_columns(
        school_df,
        [division_col, match_key, school_lat_col, school_lon_col],
        "School CSV",
    )
    validate_required_columns(
        sdo_df,
        [match_key, sdo_lat_col, sdo_lon_col],
        "SDO CSV",
    )

    school_df[school_lat_col] = pd.to_numeric(school_df[school_lat_col], errors="coerce")
    school_df[school_lon_col] = pd.to_numeric(school_df[school_lon_col], errors="coerce")
    sdo_df[sdo_lat_col] = pd.to_numeric(sdo_df[sdo_lat_col], errors="coerce")
    sdo_df[sdo_lon_col] = pd.to_numeric(sdo_df[sdo_lon_col], errors="coerce")

    duplicate_sdo_rows = int(sdo_df.duplicated(subset=[match_key], keep=False).sum())
    duplicate_sdo_keys = int(
        sdo_df.loc[sdo_df.duplicated(subset=[match_key], keep=False), match_key]
        .nunique(dropna=True)
    )

    if duplicate_sdo_rows:
        print(
            f"Warning: SDO CSV has {duplicate_sdo_rows} rows across "
            f"{duplicate_sdo_keys} duplicated match_key groups."
        )
        print("Keeping the first SDO row for each duplicated match_key.")

    sdo_lookup = (
        sdo_df[[match_key, sdo_lat_col, sdo_lon_col]]
        .drop_duplicates(subset=[match_key], keep="first")
        .rename(
            columns={
                sdo_lat_col: "_sdo_latitude",
                sdo_lon_col: "_sdo_longitude",
            }
        )
    )

    merged_df = school_df.merge(sdo_lookup, on=match_key, how="left")

    valid_distance_rows = (
        merged_df[school_lat_col].notna()
        & merged_df[school_lon_col].notna()
        & merged_df["_sdo_latitude"].notna()
        & merged_df["_sdo_longitude"].notna()
    )

    merged_df[distance_col] = pd.NA
    merged_df.loc[valid_distance_rows, distance_col] = haversine_distance_km(
        merged_df.loc[valid_distance_rows, school_lat_col],
        merged_df.loc[valid_distance_rows, school_lon_col],
        merged_df.loc[valid_distance_rows, "_sdo_latitude"],
        merged_df.loc[valid_distance_rows, "_sdo_longitude"],
    ).round(6)

    merged_df = add_extreme_outlier_columns(
        merged_df,
        division_col=division_col,
        school_lat_col=school_lat_col,
        school_lon_col=school_lon_col,
        distance_col=distance_col,
        outlier_percentile_col=outlier_percentile_col,
        nearby_count_col=nearby_count_col,
        outlier_col=outlier_col,
        nearby_radius_km=nearby_radius_km,
    )

    output_df = merged_df[
        original_school_columns
        + [distance_col, outlier_percentile_col, nearby_count_col, outlier_col]
    ].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print("Done.")
    print(f"School rows processed: {len(output_df):,}")
    print(f"Distances computed: {int(valid_distance_rows.sum()):,}")
    print(f"Blank distances: {int((~valid_distance_rows).sum()):,}")
    print(f"Nearby radius used: {nearby_radius_km} km")
    print("Outlier category counts:")
    for category, count in output_df[outlier_col].value_counts().sort_index().items():
        print(f"  {category}: {count:,}")
    print(f"Output saved to: {output_csv}")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute school-to-SDO distance and flag 99th-percentile outliers."
    )
    parser.add_argument("--school-csv", default="ph_schools_202606121415.csv")
    parser.add_argument("--sdo-csv", default="sdo_classification_202606151013.csv")
    parser.add_argument("--output-csv", default="ph_schools_with_distance_and_outliers.csv")
    parser.add_argument("--match-key", default="match_key")
    parser.add_argument("--division-col", default="division")
    parser.add_argument("--school-lat-col", default="latitude")
    parser.add_argument("--school-lon-col", default="longitude")
    parser.add_argument("--sdo-lat-col", default="latitude")
    parser.add_argument("--sdo-lon-col", default="longitude")
    parser.add_argument("--distance-col", default="linear_distance_km")
    parser.add_argument("--outlier-percentile-col", default="distance_outlier_percentile")
    parser.add_argument("--nearby-count-col", default="nearby_school_count")
    parser.add_argument("--outlier-col", default="coordinate_outlier_category")
    parser.add_argument(
        "--nearby-radius-km",
        type=float,
        default=5,
        help="Distance radius used to count nearby schools in the same division.",
    )

    args = parser.parse_args()

    add_school_to_sdo_distance(
        school_csv=args.school_csv,
        sdo_csv=args.sdo_csv,
        output_csv=args.output_csv,
        match_key=args.match_key,
        division_col=args.division_col,
        school_lat_col=args.school_lat_col,
        school_lon_col=args.school_lon_col,
        sdo_lat_col=args.sdo_lat_col,
        sdo_lon_col=args.sdo_lon_col,
        distance_col=args.distance_col,
        outlier_percentile_col=args.outlier_percentile_col,
        nearby_count_col=args.nearby_count_col,
        outlier_col=args.outlier_col,
        nearby_radius_km=args.nearby_radius_km,
    )


# Run:
# python distance_py.py
#
# Output:
# ph_schools_with_distance_and_outliers.csv
#
# Output columns:
# - all original columns from ph_schools_202606121415.csv
# - linear_distance_km
# - distance_outlier_percentile
# - nearby_school_count
# - coordinate_outlier_category
#
# distance_outlier_percentile:
# - 50 = middle of the division by distance
# - 90 = farther than about 90% of schools in the same division
# - 99+ = extreme distance outlier candidate
#
# nearby_school_count:
# - number of other schools in the same division within 5 km by default
# - diagnostic context only; it does not prevent an outlier tag
#
# coordinate_outlier_category values:
# - normal
# - extreme_outlier
# - missing_distance
# - insufficient_division_data
#
# Optional:
# Change the nearby radius:
# python distance_py.py --nearby-radius-km 10