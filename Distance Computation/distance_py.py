import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class Progress:
    def __init__(self):
        self.last_pct = -1

    def update(self, pct, label):
        pct = max(0, min(100, float(pct)))
        if int(pct) != self.last_pct:
            self.last_pct = int(pct)
            print(f"\r[{pct:6.2f}%] {label}", end="", flush=True)

    def done(self, label="Done"):
        self.update(100, label)
        print("", flush=True)


def phase_pct(start, end, current, total):
    if total <= 0:
        return end
    return start + ((end - start) * current / total)


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


def nearby_counts_chunked(lats, lons, radius_km=5, chunk_size=1000):
    """
    Counts other schools within radius_km using chunked vectorized distance matrices.

    This is much faster than looping row-by-row, while avoiding one massive
    all-pairs matrix for large divisions.
    """
    row_count = len(lats)
    counts = np.zeros(row_count, dtype=np.int32)

    for start in range(0, row_count, chunk_size):
        end = min(start + chunk_size, row_count)

        distance_matrix = haversine_distance_km(
            lats[start:end, None],
            lons[start:end, None],
            lats[None, :],
            lons[None, :],
        )

        # Subtract 1 to exclude the school itself.
        counts[start:end] = (distance_matrix <= radius_km).sum(axis=1) - 1

    return counts


def fast_spatial_fallback_scores(features):
    """
    Fast fallback when scikit-learn is not installed.

    Higher score = more anomalous. This is not a true Isolation Forest; it is a
    standardized distance-from-center score used to avoid the very slow
    pure-Python tree implementation.
    """
    medians = np.median(features, axis=0)
    mad = np.median(np.abs(features - medians), axis=0)
    mad = np.where(mad == 0, 1, mad)
    robust_z = (features - medians) / mad
    return np.sqrt((robust_z**2).sum(axis=1))


def add_outlier_columns(
    df,
    progress,
    division_col="division",
    school_lat_col="latitude",
    school_lon_col="longitude",
    distance_col="linear_distance_km",
    outlier_percentile_col="distance_outlier_percentile",
    nearby_count_col="nearby_school_count",
    outlier_col="coordinate_outlier_category",
    isolation_forest_score_col="isolation_forest_score",
    isolation_forest_outlier_col="isolation_forest_outlier",
    outlier_method_col="coordinate_outlier_method",
    min_schools_per_division=5,
    min_schools_per_isolation_forest=10,
    extreme_percentile=99,
    nearby_radius_km=5,
    nearby_chunk_size=1000,
    isolation_contamination=0.02,
    isolation_random_state=42,
    compute_nearby_counts=True,
    run_isolation_forest=True,
):
    df[outlier_percentile_col] = pd.NA
    df[nearby_count_col] = pd.NA
    df[isolation_forest_score_col] = pd.NA
    df[isolation_forest_outlier_col] = False
    df[outlier_method_col] = "none"
    df[outlier_col] = "normal"

    valid_coordinates = df[school_lat_col].notna() & df[school_lon_col].notna()
    coordinate_groups = list(df.loc[valid_coordinates].groupby(division_col, dropna=False))

    if compute_nearby_counts:
        for group_number, (division, group) in enumerate(coordinate_groups, start=1):
            progress.update(
                phase_pct(25, 50, group_number - 1, len(coordinate_groups)),
                f"Counting nearby schools: {group_number}/{len(coordinate_groups)} divisions",
            )

            lats = group[school_lat_col].astype(float).to_numpy()
            lons = group[school_lon_col].astype(float).to_numpy()

            df.loc[group.index, nearby_count_col] = nearby_counts_chunked(
                lats,
                lons,
                radius_km=nearby_radius_km,
                chunk_size=nearby_chunk_size,
            )
    else:
        progress.update(50, "Skipped nearby-school counts")

    missing_distance = df[distance_col].isna()
    df.loc[missing_distance, outlier_col] = "missing_distance"
    df.loc[missing_distance, outlier_method_col] = "missing_distance"

    usable_groups = list(df.loc[~missing_distance].groupby(division_col, dropna=False))

    for group_number, (division, group) in enumerate(usable_groups, start=1):
        progress.update(
            phase_pct(50, 65, group_number - 1, len(usable_groups)),
            f"Ranking distance percentiles: {group_number}/{len(usable_groups)} divisions",
        )

        if len(group) < min_schools_per_division:
            df.loc[group.index, outlier_col] = "insufficient_division_data"
            df.loc[group.index, outlier_method_col] = "insufficient_division_data"
            continue

        percentiles = (
            group[distance_col]
            .astype(float)
            .rank(method="max", pct=True)
            .mul(100)
            .round(2)
        )

        df.loc[group.index, outlier_percentile_col] = percentiles

        extreme_indexes = percentiles[percentiles >= extreme_percentile].index
        df.loc[extreme_indexes, outlier_col] = "extreme_outlier"
        df.loc[extreme_indexes, outlier_method_col] = "univariate_percentile"

    isolation_rows = df.loc[
        df[school_lat_col].notna()
        & df[school_lon_col].notna()
        & df[distance_col].notna()
    ]
    isolation_groups = list(isolation_rows.groupby(division_col, dropna=False))

    if run_isolation_forest:
        if not HAS_SKLEARN:
            print(
                "\nWarning: scikit-learn is not installed. "
                "Using fast spatial fallback scores instead of true Isolation Forest."
            )

        for group_number, (division, group) in enumerate(isolation_groups, start=1):
            progress.update(
                phase_pct(65, 90, group_number - 1, len(isolation_groups)),
                f"Scoring spatial outliers: {group_number}/{len(isolation_groups)} divisions",
            )

            if len(group) < min_schools_per_isolation_forest:
                continue

            features = (
                group[[school_lat_col, school_lon_col, distance_col]]
                .astype(float)
                .to_numpy()
            )

            feature_means = features.mean(axis=0)
            feature_stds = features.std(axis=0)
            feature_stds = np.where(feature_stds == 0, 1, feature_stds)
            normalized_features = (features - feature_means) / feature_stds

            outlier_count = max(1, int(np.ceil(len(group) * isolation_contamination)))

            if HAS_SKLEARN:
                model = IsolationForest(
                    n_estimators=100,
                    max_samples=min(256, len(group)),
                    contamination=isolation_contamination,
                    random_state=isolation_random_state,
                    n_jobs=-1,
                )
                model.fit(normalized_features)
                scores = -model.score_samples(normalized_features)
            else:
                scores = fast_spatial_fallback_scores(normalized_features)

            df.loc[group.index, isolation_forest_score_col] = np.round(scores, 6)

            score_cutoff = np.sort(scores)[-outlier_count]
            isolation_indexes = group.index[scores >= score_cutoff]
            df.loc[isolation_indexes, isolation_forest_outlier_col] = True

            existing_extreme_indexes = df.loc[isolation_indexes].index[
                df.loc[isolation_indexes, outlier_col].eq("extreme_outlier")
            ]
            df.loc[
                existing_extreme_indexes,
                outlier_method_col,
            ] = "univariate_percentile+isolation_forest"

            new_isolation_indexes = df.loc[isolation_indexes].index[
                df.loc[isolation_indexes, outlier_col].eq("normal")
            ]
            df.loc[new_isolation_indexes, outlier_col] = "isolation_forest_outlier"
            df.loc[new_isolation_indexes, outlier_method_col] = "isolation_forest"
    else:
        progress.update(90, "Skipped spatial outlier scoring")

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
    isolation_forest_score_col="isolation_forest_score",
    isolation_forest_outlier_col="isolation_forest_outlier",
    outlier_method_col="coordinate_outlier_method",
    nearby_radius_km=5,
    nearby_chunk_size=1000,
    isolation_contamination=0.02,
    isolation_random_state=42,
    compute_nearby_counts=True,
    run_isolation_forest=True,
):
    """
    Creates a new CSV with:
    - all original columns from ph_schools
    - linear_distance_km
    - distance_outlier_percentile
    - nearby_school_count
    - isolation_forest_score
    - isolation_forest_outlier
    - coordinate_outlier_category
    - coordinate_outlier_method

    Speed improvements:
    - vectorized distance calculation
    - chunked vectorized nearby-school counts
    - scikit-learn IsolationForest with n_jobs=-1 when installed
    - fast fallback if scikit-learn is unavailable
    - console percentage progress indicator
    """

    progress = Progress()

    school_csv = Path(school_csv)
    sdo_csv = Path(sdo_csv)
    output_csv = Path(output_csv)

    progress.update(1, "Loading CSV files")
    school_df = pd.read_csv(school_csv, low_memory=False)
    sdo_df = pd.read_csv(sdo_csv, low_memory=False)

    original_school_columns = list(school_df.columns)

    progress.update(8, "Validating columns")
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
            f"\nWarning: SDO CSV has {duplicate_sdo_rows} rows across "
            f"{duplicate_sdo_keys} duplicated match_key groups."
        )
        print("Keeping the first SDO row for each duplicated match_key.")

    progress.update(15, "Merging schools with SDO coordinates")
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

    progress.update(20, "Computing school-to-SDO distances")
    valid_distance_rows = (
        merged_df[school_lat_col].notna()
        & merged_df[school_lon_col].notna()
        & merged_df["_sdo_latitude"].notna()
        & merged_df["_sdo_longitude"].notna()
    )

    merged_df[distance_col] = pd.NA
    merged_df.loc[valid_distance_rows, distance_col] = haversine_distance_km(
        merged_df.loc[valid_distance_rows, school_lat_col].astype(float).to_numpy(),
        merged_df.loc[valid_distance_rows, school_lon_col].astype(float).to_numpy(),
        merged_df.loc[valid_distance_rows, "_sdo_latitude"].astype(float).to_numpy(),
        merged_df.loc[valid_distance_rows, "_sdo_longitude"].astype(float).to_numpy(),
    ).round(6)

    merged_df = add_outlier_columns(
        merged_df,
        progress=progress,
        division_col=division_col,
        school_lat_col=school_lat_col,
        school_lon_col=school_lon_col,
        distance_col=distance_col,
        outlier_percentile_col=outlier_percentile_col,
        nearby_count_col=nearby_count_col,
        outlier_col=outlier_col,
        isolation_forest_score_col=isolation_forest_score_col,
        isolation_forest_outlier_col=isolation_forest_outlier_col,
        outlier_method_col=outlier_method_col,
        nearby_radius_km=nearby_radius_km,
        nearby_chunk_size=nearby_chunk_size,
        isolation_contamination=isolation_contamination,
        isolation_random_state=isolation_random_state,
        compute_nearby_counts=compute_nearby_counts,
        run_isolation_forest=run_isolation_forest,
    )

    progress.update(92, "Preparing output columns")
    output_df = merged_df[
        original_school_columns
        + [
            distance_col,
            outlier_percentile_col,
            nearby_count_col,
            isolation_forest_score_col,
            isolation_forest_outlier_col,
            outlier_col,
            outlier_method_col,
        ]
    ].copy()

    progress.update(96, "Writing output CSV")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    progress.done("Complete")

    print(f"School rows processed: {len(output_df):,}")
    print(f"Distances computed: {int(valid_distance_rows.sum()):,}")
    print(f"Blank distances: {int((~valid_distance_rows).sum()):,}")
    print(f"Nearby radius used: {nearby_radius_km} km")
    print(f"Nearby counts computed: {compute_nearby_counts}")
    print(f"Spatial outlier scoring enabled: {run_isolation_forest}")
    print(f"scikit-learn IsolationForest used: {HAS_SKLEARN and run_isolation_forest}")
    print(f"Isolation contamination: {isolation_contamination}")
    print(
        "Spatial outliers: "
        f"{int(output_df[isolation_forest_outlier_col].sum()):,}"
    )
    print("Outlier category counts:")
    for category, count in output_df[outlier_col].value_counts().sort_index().items():
        print(f"  {category}: {count:,}")
    print(f"Output saved to: {output_csv}")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute school-to-SDO distance and flag coordinate outliers."
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
    parser.add_argument("--isolation-forest-score-col", default="isolation_forest_score")
    parser.add_argument("--isolation-forest-outlier-col", default="isolation_forest_outlier")
    parser.add_argument("--outlier-method-col", default="coordinate_outlier_method")
    parser.add_argument(
        "--nearby-radius-km",
        type=float,
        default=5,
        help="Distance radius used to count nearby schools in the same division.",
    )
    parser.add_argument(
        "--nearby-chunk-size",
        type=int,
        default=1000,
        help="Rows per chunk for nearby-school distance matrices.",
    )
    parser.add_argument(
        "--isolation-contamination",
        type=float,
        default=0.02,
        help="Expected share of spatial outliers within each eligible division.",
    )
    parser.add_argument(
        "--isolation-random-state",
        type=int,
        default=42,
        help="Random seed for reproducible spatial outlier scoring.",
    )
    parser.add_argument(
        "--no-nearby-counts",
        action="store_true",
        help="Skip nearby_school_count to finish faster.",
    )
    parser.add_argument(
        "--skip-isolation-forest",
        action="store_true",
        help="Skip spatial outlier scoring to finish faster.",
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
        isolation_forest_score_col=args.isolation_forest_score_col,
        isolation_forest_outlier_col=args.isolation_forest_outlier_col,
        outlier_method_col=args.outlier_method_col,
        nearby_radius_km=args.nearby_radius_km,
        nearby_chunk_size=args.nearby_chunk_size,
        isolation_contamination=args.isolation_contamination,
        isolation_random_state=args.isolation_random_state,
        compute_nearby_counts=not args.no_nearby_counts,
        run_isolation_forest=not args.skip_isolation_forest,
    )


# Run:
# python distance_py.py
#
# Fastest run, while keeping distance + percentile outliers:
# python distance_py.py --no-nearby-counts --skip-isolation-forest
#
# Output:
# ph_schools_with_distance_and_outliers.csv
#
# Output columns:
# - all original columns from ph_schools_202606121415.csv
# - linear_distance_km
# - distance_outlier_percentile
# - nearby_school_count
# - isolation_forest_score
# - isolation_forest_outlier
# - coordinate_outlier_category
# - coordinate_outlier_method
#
# distance_outlier_percentile:
# - 50 = middle of the division by distance
# - 90 = farther than about 90% of schools in the same division
# - 99+ = extreme distance outlier candidate
#
# nearby_school_count:
# - number of other schools in the same division within 5 km by default
# - diagnostic context only; it does not prevent an outlier tag
# - use --no-nearby-counts if this diagnostic step is slowing the run
#
# isolation_forest_score:
# - higher means more anomalous based on latitude, longitude, and linear distance
# - uses scikit-learn IsolationForest when installed
# - otherwise uses a fast spatial fallback to avoid the slow pure-Python tree loop
#
# isolation_forest_outlier:
# - True when the school is separated from the division majority according to
#   spatial outlier scoring
#
# coordinate_outlier_category values:
# - normal
# - extreme_outlier
# - isolation_forest_outlier
# - missing_distance
# - insufficient_division_data
#
# coordinate_outlier_method values:
# - none
# - univariate_percentile
# - isolation_forest
# - univariate_percentile+isolation_forest
# - missing_distance
# - insufficient_division_data
#
# Optional:
# Change the nearby radius:
# python distance_py.py --nearby-radius-km 10
#
# Tune spatial outlier sensitivity:
# python distance_py.py --isolation-contamination 0.03