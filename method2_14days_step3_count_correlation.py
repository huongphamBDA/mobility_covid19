# This script continues the script get_df_dots_14days to count the days that showed correlation b/w case and flows in
# 41 San Diego SRAs from 2020-04-01 to 2021-03-01

import os
import pandas
import geopandas
from matplotlib import pyplot as plt


def main():
    df = pandas.read_csv("output_method2_14days/df_count_correlation.csv")

    # ======
    # Find out SRA names for final conclusion
    list_sra_in = df[df["inflow"] > 15].sort_values(by=["inflow"], ascending=False)["SRA names"]
    list_sra_net = df[df["netflow"] > 15].sort_values(by=["netflow"], ascending=False)["SRA names"]
    list_sra_out = df[df["outflow"] > 15].sort_values(by=["outflow"], ascending=False)["SRA names"]
    list_sra_inwithin = df[df["inwithinflow"] > 15].sort_values(by=["inwithinflow"], ascending=False)["SRA names"]
    list_sra_within = df[df["withinflow"] > 15].sort_values(by=["withinflow"], ascending=False)["SRA names"]

    all_lists = [list_sra_in, list_sra_net, list_sra_out, list_sra_inwithin, list_sra_within]

    print("\nSRAs having case correlated (increase - increase relationship) with inflow: ", *list_sra_in)
    print("\nSRAs having case correlated (increase - increase relationship) with netflow: ", *list_sra_net)
    print("\nSRAs having case correlated (increase - increase relationship) with outflow: ", *list_sra_out)
    print("\nSRAs having case correlated (increase - increase relationship) with inwithinflow: ", *list_sra_inwithin)
    print("\nSRAs having case correlated (increase - increase relationship) with withinflow: ", *list_sra_within)

    mutual_sras = set.intersection(*[set(alist) for alist in all_lists])  # La Mesa, National City
    print("\nSRAs having case correlated (increase - increase relationship) with all types of flows: ", mutual_sras)

    # ======
    # get stats for DTW values
    column_names=["Human mobility flow", "max_dtw", "min_dtw", "mean_dtw", "standard deviation_dtw"]
    max, min, mean, std = [], [], [], []

    flows = ["inflow", "netflow", "outflow", "inwithinflow", "withinflow"]
    for flow in flows:
        max.append(df[flow].max())
        min.append(df[flow].min())
        mean.append(round(df[flow].mean(), 2))
        std.append(round(df[flow].std(), 2))

    big_list = list(zip(flows, max, min, mean, std))
    df_dtw_stats = pandas.DataFrame(big_list, columns=column_names)
    df_dtw_stats.to_csv("./output_method2_14days/df_dtw_stats.csv")

    # ======
    # create choropleth map
    sra_shapefile = geopandas.read_file("./data/sra2000/sra2000.shp")
    sra_shapefile['NAME'] = sra_shapefile['NAME'].str.upper()

    # Get coords for geometry centroids
    sra_shapefile['geom_centroid'] = sra_shapefile.centroid
    sra_shapefile['coords'] = sra_shapefile['geom_centroid'].apply(lambda x: x.representative_point().coords[:])
    sra_shapefile['coords'] = [coords[0] for coords in sra_shapefile['coords']]

    df_merged = sra_shapefile.merge(df, how='left', left_on=['NAME'], right_on=['SRA names'])
    df_merged = df_merged.drop(df_merged.columns[3], axis=1)  # drop column SRA_y (column index 3)

    gdf_dtw = geopandas.GeoDataFrame(df_merged, geometry=df_merged["geometry"])
    gdf_dtw.set_crs(epsg=4326, allow_override=True)

    flows = ["inflow", "netflow", "outflow", "inwithinflow", "withinflow"]

    for flow in flows:
        gdf_dtw.loc[gdf_dtw[flow] <= 15, flow] = 0
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'aspect': 'equal'})
        gdf_dtw.plot(column=flow, legend=True, ax=ax, cmap="Blues")

        # Label SRA names in the choropleth maps
        for ind, row in gdf_dtw.iterrows():
            plt.annotate(text=row["NAME"], xy=row["coords"], horizontalalignment="center", fontsize=5)

        # https://geopandas.org/en/stable/docs/user_guide/mapping.html
        plt.title(f"Correlation Count of 41 SRAs, 14 days approach -- {flow}")

        # Save file
        fn = f"choropleth_count_correlation_{flow}.png"
        fp = os.path.join("output_method2_14days/_choropleth_count_correlation_14days", fn)
        plt.savefig(fp, bbox_inches="tight")

        # plt.show()


if __name__ == "__main__":
    main()


