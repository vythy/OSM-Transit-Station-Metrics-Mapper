import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx


PLACE = "Boston, Massachusetts, USA"
BUFFER_METERS = 0.5 * 1609.34
OUTPUT_CSV = "transit_station_metrics.csv"


def safe_representative_point(geom):
    if geom is None or geom.is_empty:
        return None
    return geom.representative_point()


def clean_geometries(gdf):
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def get_station_name_columns(gdf):
    possible_cols = ["name", "ref", "railway", "public_transport", "highway"]
    for col in possible_cols:
        if col in gdf.columns:
            return col
    return None

def classify_mbta_station(row):
    railway = row.get("railway")
    station = row.get("station")
    highway = row.get("highway")

    if highway == "bus_stop":
        return pd.Series([False, "not_mbta_tod"])

    if railway == "subway_entrance":
        return pd.Series([False, "not_mbta_tod"])

    if station == "subway":
        return pd.Series([True, "rapid_transit"])

    if station == "light_rail":
        return pd.Series([True, "light_rail"])

    if station == "train":
        return pd.Series([True, "commuter_rail"])

    if railway in ["station", "halt"]:
        return pd.Series([True, "rail_station_unspecified"])

    return pd.Series([False, "not_mbta_tod"])

def download_data(place):
    print("Downloading walking network...")
    G = ox.graph_from_place(place, network_type="walk")

    print("Downloading transit features...")
    transit = ox.features_from_place(
        place,
        tags={
            "public_transport": True,
            "railway": ["station", "halt", "tram_stop", "subway_entrance"],
            "highway": "bus_stop",
        },
    )

    print("Downloading buildings...")
    buildings = ox.features_from_place(place, tags={"building": True})

    print("Downloading schools...")
    schools = ox.features_from_place(place, tags={"amenity": "school"})

    print("Downloading hospitals...")
    hospitals = ox.features_from_place(place, tags={"amenity": "hospital"})

    print("Downloading parks...")
    parks = ox.features_from_place(place, tags={"leisure": "park"})

    return G, transit, buildings, schools, hospitals, parks


def prepare_data(G, transit, buildings, schools, hospitals, parks):
    print("Cleaning geometries...")
    transit = clean_geometries(transit)
    buildings = clean_geometries(buildings)
    schools = clean_geometries(schools)
    hospitals = clean_geometries(hospitals)
    parks = clean_geometries(parks)

    print("Converting point-like layers to representative points...")
    transit["geometry"] = transit.geometry.apply(safe_representative_point)
    schools["geometry"] = schools.geometry.apply(safe_representative_point)
    hospitals["geometry"] = hospitals.geometry.apply(safe_representative_point)

    transit = clean_geometries(transit)
    schools = clean_geometries(schools)
    hospitals = clean_geometries(hospitals)

    print("Projecting graph and layers...")

    # project graph
    G_proj = ox.project_graph(G)

    # get CRS of projected graph
    nodes, edges = ox.graph_to_gdfs(G_proj)
    target_crs = nodes.crs

    # project all layers to same CRS
    transit = transit.to_crs(target_crs)
    buildings = buildings.to_crs(target_crs)
    schools = schools.to_crs(target_crs)
    hospitals = hospitals.to_crs(target_crs)
    parks = parks.to_crs(target_crs)

    print("Creating station table...")
    transit = transit.reset_index(drop=False).copy()
    transit["station_id"] = range(len(transit))

    name_col = get_station_name_columns(transit)
    if name_col is not None:
        transit["station_name"] = transit[name_col].astype(str)
        transit.loc[transit["station_name"].isin(["nan", "None"]), "station_name"] = None
    else:
        transit["station_name"] = None

    transit["station_name"] = transit["station_name"].fillna(
        "station_" + transit["station_id"].astype(str)
    )

    transit[["is_mbta_tod_station", "mbta_mode"]] = transit.apply(
        classify_mbta_station, axis=1
    )

    return G_proj, transit, buildings, schools, hospitals, parks


def count_features_in_buffers(station_buffers, features_gdf, output_col):
    if len(features_gdf) == 0:
        return pd.DataFrame(
            {
                "station_id": station_buffers["station_id"],
                output_col: 0,
            }
        )

    joined = gpd.sjoin(
        features_gdf[["geometry"]],
        station_buffers[["station_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    counts = (
        joined.groupby("station_id")
        .size()
        .rename(output_col)
        .reset_index()
    )

    return counts


def compute_avg_walk_distance(G_proj, transit, buildings, station_buffers):
    print("Snapping stations to network nodes...")
    transit["node_id"] = ox.nearest_nodes(
        G_proj,
        X=transit.geometry.x,
        Y=transit.geometry.y,
    )

    print("Preparing building representative points...")
    building_points = buildings.copy()
    building_points["geometry"] = building_points.geometry.apply(safe_representative_point)
    building_points = clean_geometries(building_points)

    print("Snapping building points to network nodes...")
    building_points["node_id"] = ox.nearest_nodes(
        G_proj,
        X=building_points.geometry.x,
        Y=building_points.geometry.y,
    )

    print("Joining buildings to station buffers...")
    bp_join = gpd.sjoin(
        building_points[["geometry", "node_id"]],
        station_buffers[["station_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    station_node_lookup = transit.set_index("station_id")["node_id"].to_dict()

    print("Computing average walking distances...")
    avg_dist_rows = []

    for station_id, group in bp_join.groupby("station_id"):
        station_node = station_node_lookup[station_id]
        distances = []

        unique_building_nodes = group["node_id"].dropna().unique()

        for building_node in unique_building_nodes:
            try:
                dist = nx.shortest_path_length(
                    G_proj,
                    source=building_node,
                    target=station_node,
                    weight="length",
                )
                distances.append(dist)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        avg_dist = sum(distances) / len(distances) if distances else None

        avg_dist_rows.append(
            {
                "station_id": station_id,
                "avg_walk_distance_m": avg_dist,
            }
        )

    avg_dist_df = pd.DataFrame(avg_dist_rows)

    if avg_dist_df.empty:
        avg_dist_df = pd.DataFrame(
            {
                "station_id": transit["station_id"],
                "avg_walk_distance_m": None,
            }
        )

    return avg_dist_df


def main():
    G, transit, buildings, schools, hospitals, parks = download_data(PLACE)

    G_proj, transit, buildings, schools, hospitals, parks = prepare_data(
        G, transit, buildings, schools, hospitals, parks
    )

    print("Creating station buffers...")
    station_buffers = transit[["station_id", "station_name", "geometry"]].copy()
    station_buffers["geometry"] = station_buffers.geometry.buffer(BUFFER_METERS)

    print("Counting buildings...")
    building_counts = count_features_in_buffers(
        station_buffers, buildings, "buildings"
    )

    print("Counting schools...")
    school_counts = count_features_in_buffers(
        station_buffers, schools, "schools"
    )

    print("Counting hospitals...")
    hospital_counts = count_features_in_buffers(
        station_buffers, hospitals, "hospitals"
    )

    print("Counting parks...")
    park_counts = count_features_in_buffers(
        station_buffers, parks, "parks"
    )

    avg_dist_df = compute_avg_walk_distance(
        G_proj, transit, buildings, station_buffers
    )

    df = transit[
        ["station_id", "station_name", "is_mbta_tod_station", "mbta_mode"]
    ].copy()

    for counts in [building_counts, school_counts, hospital_counts, park_counts, avg_dist_df]:
        df = df.merge(counts, on="station_id", how="left")

    df["buildings"] = df["buildings"].fillna(0).astype(int)
    df["schools"] = df["schools"].fillna(0).astype(int)
    df["hospitals"] = df["hospitals"].fillna(0).astype(int)
    df["parks"] = df["parks"].fillna(0).astype(int)

    df = df.sort_values("station_name").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()