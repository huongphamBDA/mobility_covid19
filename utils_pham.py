# viz_oct_data_overall and all "oct_data" is because the data was updated on 10/01/2022
import os
import pandas
import plotly.graph_objects as go
import plotly.express as px
import geopandas
import math
import numpy
import scipy
import warnings
import time

from scipy import stats
from matplotlib import pyplot as plt
from tslearn.metrics import dtw
from plotly.subplots import make_subplots
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def create_output_folders():
    out_dir1_1 = "viz_oct_data_overall/df_min_max/one_metric_one_sra_oct_data"
    out_dir1_2 = "./viz_oct_data_overall/df_min_max/one_metric_all_sra_oct_data"
    out_dir1_3 = "./viz_oct_data_overall/df_min_max/multiple_metric_one_sra_oct_data"
    out_dir2_1 = "./output_method1_phases/1_dtw"
    out_dir3_1 = "./output_method1_phases/2_clusters/1_cluster_1plot"  # show each cluster in k clusters
    out_dir3_2 = "./output_method1_phases/2_clusters/cluster_distribution"
    out_dir3_3 = "./output_method1_phases/2_clusters/k_clusters_1plot"
    out_dir4_1 = "./output_method1_phases/3_results/results"
    out_dir5_1_html = "./output_method1_phases/3_results/categories/_increase_table"
    out_dir5_2_html = "./output_method1_phases/3_results/categories/_stable_table"
    out_dir5_3_html = "./output_method1_phases/3_results/categories/_decrease_table"
    out_dir5_4_html = "./output_method1_phases/3_results/categories/_outlier_table"
    out_dir5_5_html = "./output_method1_phases/3_results/conclusions"

    outdir_14days = "./output_method2_14days"

    for folder in [out_dir1_1, out_dir1_2, out_dir1_3,
                   out_dir2_1,
                   out_dir3_1, out_dir3_2, out_dir3_3,
                   out_dir4_1,
                   out_dir5_1_html, out_dir5_2_html, out_dir5_3_html, out_dir5_4_html, out_dir5_5_html,
                   outdir_14days]:

        if not os.path.exists(folder):
            os.makedirs(folder)

    return out_dir1_1, out_dir1_2, out_dir1_3, out_dir2_1, out_dir3_1, out_dir3_2, out_dir3_3, out_dir4_1, \
        out_dir5_1_html, out_dir5_2_html, out_dir5_3_html, out_dir5_4_html, out_dir5_5_html, outdir_14days


# used by both methods
def define_flow_and_case_cols_and_phases():
    flow_cols = ["inflow_weekly_avg", "outflow_weekly_avg", "withinflow_weekly_avg", "netflow_weekly_avg",
                 "total_in_within_weekly_avg"]
    flow_cols_short = ["in", "out", "within", "net", "inwithin"]
    case_n_flow_cols = ["cases_weekly_avg", "inflow_weekly_avg", "outflow_weekly_avg", "withinflow_weekly_avg",
                        "netflow_weekly_avg", "total_in_within_weekly_avg"]

    cols_to_viz = ['cases_weekly_avg', "case_acum",
                   "inflow", "inflow_weekly_avg", "in_sd_weekly_avg", "in_non_sd_weekly_avg",
                   "outflow", "outflow_weekly_avg", "out_sd_weekly_avg", "out_non_sd_weekly_avg",
                   "netflow", 'netflow_weekly_avg', 'netflow_sd_weekly_avg', 'netflow_non_sd_weekly_avg',
                   "withinflow", "withinflow_weekly_avg",
                   "total_in_within", "total_in_within_weekly_avg"]

    # phase_names = ["phase1", "phase2", "phase3", "phase4", "phase5"] # edit
    phase_names = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6"]

    # phase_dates = [['2020-05-01', '2020-6-24'], ['2020-06-25', '2020-8-18'], ['2020-08-19', '2020-10-31'],
    #                ['2020-11-01', '2021-1-23'], ['2021-01-24', '2021-4-3']]  # edit
    phase_dates = [['2020-04-01', '2020-05-31'], ['2020-06-01', '2020-07-20'], ['2020-07-21', '2020-08-31'],
                   ['2020-09-01', '2020-11-05'], ['2020-11-06', '2021-01-05'], ['2021-01-06', '2021-03-27']]

    return case_n_flow_cols, flow_cols, flow_cols_short, cols_to_viz, phase_dates, phase_names


def print_heading(title):
    print("\n")
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def print_subheading(title):
    print("\n")
    print("-" * 50)
    print(title)
    print("-" * 50)
    return


def cal_roll_avg(array, num_day, min_obs):
    """Calculate seven day moving average (= rolling average)"""

    return array.rolling(window=num_day, min_periods=min_obs).mean()


def divide_time(df):
    """Divide time into phases.
    Return a list of arrays; each array is an array of True or False values"""
    # timeframe =[ [start-date1, eenddate1], ['2020-04-01', '2020-04-30'], [,]]

    # phase1 = (df['date'] >= '2020-05-01') & (df['date'] <= '2020-6-24')
    # phase2 = (df['date'] >= '2020-06-25') & (df['date'] <= '2020-8-18')
    # phase3 = (df['date'] >= '2020-08-19') & (df['date'] <= '2020-10-31')
    # phase4 = (df['date'] >= '2020-11-01') & (df['date'] <= '2021-1-23')
    # phase5 = (df['date'] >= '2021-01-24') & (df['date'] <= '2021-4-3')  # edit

    phase1 = (df['date'] >= '2020-04-01') & (df['date'] <= '2020-05-31')
    phase2 = (df['date'] >= '2020-06-01') & (df['date'] <= '2020-07-20')
    phase3 = (df['date'] >= '2020-07-21') & (df['date'] <= '2020-08-31')
    phase4 = (df['date'] >= '2020-09-01') & (df['date'] <= '2020-11-05')
    phase5 = (df['date'] >= '2020-11-06') & (df['date'] <= '2021-01-05')
    phase6 = (df['date'] >= '2021-01-06') & (df['date'] <= '2021-03-27')

    # phase_TF_list = [phase1, phase2, phase3, phase4, phase5]
    phase_TF_list = [phase1, phase2, phase3, phase4, phase5, phase6]

    return phase_TF_list


def check_missing_rows(df, col, df_name):
    bool_val = pandas.isnull(df[col])
    df_missing = df[bool_val]

    if not df_missing.empty:
        print(f"Having {df_missing.shape[0]} missing values in {col}")
    else:
        print(f"No missing values in {col} of {df_name}")


def check_negative_values(df, col, df_name):
    df_neg = df[df[col] < 0]

    if not df_neg.empty:
        print(f"Having {df_neg.shape[0]} negative {col} in {df_name}")
    else:
        print(f"No negative {col} in {df_name}")


def nan_counter(list_of_series):
    nan_polluted_series_counter = 0

    for series in list_of_series:
        if series.isnull().sum().sum() > 0:
            nan_polluted_series_counter += 1

    print(nan_polluted_series_counter)


def find_indexes(mylist, value):
    indexes = []
    for i, v in enumerate(mylist):
        if mylist[i] == value:
            indexes.append(i)

    return indexes


def start_run_code():
    start_time = time.time()
    code_run_start = datetime.now()
    print(f"Code run starts: {code_run_start};")

    warnings.filterwarnings('ignore')  # ignore warnings

    return code_run_start, start_time


def print_code_run_time(code_run_start, start_time):
    execution_time = round((time.time() - start_time) / 60, 2)

    print(f"End time: {str(datetime.now())}; "
          f"Execution time: {str(execution_time)} minutes since {str(code_run_start)}.")


def normalize_data(df_data):
    print_subheading("\n1.4 Normalize data")

    # First way - rescale every feature to the [0, 1] interval
    df_data_minmax = df_data.copy()
    trans = MinMaxScaler()
    df_data_minmax.iloc[:, 4:] = trans.fit_transform(df_data_minmax.iloc[:, 4:])

    # Second way - convert all values to a z-score; mean value in each column is assigned a value of 0.0 and values are
    # centered around 0.0 with (+) or (-) values
    # df_data_scaled = df_data.copy()
    # scaler = StandardScaler()
    # df_data_scaled.iloc[:, 5:] = scaler.fit_transform(df_data_scaled.iloc[:, 5:])

    return df_data_minmax


def test_normal_distribution(df_data, cols):
    print_heading("2. EXPLORE AND VISUALIZE DATA")
    print_subheading("\n2.1 Test normal distribution for cases_weekly_avg")

    # p value = A 2-sided chi squared probability for the hypothesis test.
    # null hypothesis: x comes from a normal distribution
    # p < alpha: The null hypothesis can be rejected

    for col in cols:
        k, p = stats.normaltest(df_data[col])
        print(f"k, p for {col}: {k}, {p}")

    print("\nConclusion: all five arrays do not come from normal distribution")

    # Minmax normalization is better than standardization (z-score normalization) for some reasons: (i) we are not sure
    # about the data distribution; it's safer to use min_max for data across time and places like this; (ii) we might
    # need to divide data in time or places or phases (which take long time) to check normal distribution before using
    # the standardization; Machine learning estimators might behave badly if the individual features do not more or
    # less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
    # One day has 41 SRAs (geo)
    # One SRA has 400 days (time series)
    # One set has 5 phases (41 SRAs * 40 days * 5 phases)


# =========================================
#             VISUALIZE DATA
# =========================================
def plot_one_metric_each_sra(df, cols_to_viz, out_dir):
    """Visualize COVID-19 case OR human mobility flow time series for each SRA
    (each SRA, one metric, 41 small plots)"""

    for col in cols_to_viz:
        df_grouped = df.groupby('name')
        fig, axs = plt.subplots(nrows=7, ncols=6, sharex='all', sharey='all')
        grouped_sra = df_grouped.groups.keys()  # ['alpine', 'carlsbad', 'mira mesa']
        axs_array = axs.flatten()  # plot 1, plot 2, plot 3
        targets = zip(grouped_sra, axs_array)  # tuples: ('alpine', plot 1), ('carlsbad', plot 2)

        for i, (key, ax) in enumerate(targets):  # index 0, ('alpines', plot 1); index 1, ('carlsbad', plot 2)
            sra_df = df_grouped.get_group(key)  # sra_df. e.g Alpine's df, Carlsbad's df
            sra_df.plot(x='date', y=f'{col}', ax=ax, fontsize=5, figsize=(12, 10), legend=False, grid=True)
            ax.set_title(f"{key}", size=5)

        plt.suptitle(f"{col} IN 41 SAN DIEGO SRAs", fontsize=16)
        fig.delaxes(axs[6][5])

        # Save file and show plot
        fn = f"EACH_{col}.png"  # file name
        fp = os.path.join(out_dir, fn)  # file path
        plt.savefig(fp, bbox_inches="tight")  # save file to local memory

    # fig.show()
    # plt.show()


def plot_one_metric_all_sra(df, cols_to_viz, out_dir):
    """Visualize COVID-19 case or human mobility flow time series for ALL SRAs
    (all SRAs, one metric, one plot)"""

    for col in cols_to_viz:
        df_pivot = df.pivot(index="date", columns="name", values=f"{col}")
        df_pivot.plot(title=f"{col} IN 41 SAN DIEGO SRAS", ylabel=f"{col}", figsize=(12, 6), rot=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, ncol=2, shadow=True)
        plt.grid()

        fn = f"ALL_{col}.png"
        fp = os.path.join(out_dir, fn)
        plt.savefig(fp, bbox_inches="tight")

    # plt.show()


def plot_multiple_metric_one_sra(df, out_dir):
    """Case and flow time series for EACH SRA (each plot for multiple metrics of each SRA )"""

    sra_name = df['name'].unique()  # e.g. "miramar", "del mar - mira mesa" - 41 names in total

    cols_selected = ["date", "cases_weekly_avg", "inflow_weekly_avg", "outflow_weekly_avg", "netflow_weekly_avg",
                     "withinflow_weekly_avg", "total_in_within_weekly_avg"]
    color_list = ["slategrey", "crimson", "seagreen", "dodgerblue", "tomato", "mediumturquoise", "chocolate"]
    # "slategrey" will be skipped

    for sra in sra_name:
        df_sra = df.loc[df['name'] == sra, cols_selected]

        fig = go.Figure()

        for i in range(1, len(cols_selected)):
            fig.add_trace(go.Scatter(x=df_sra["date"],
                                     y=df_sra[cols_selected[i]],
                                     name=cols_selected[i],
                                     line=dict(color=color_list[i])))

        fig.update_layout(title=f'COVID-19 Case and Human Mobility Time Series in {sra}',
                          xaxis=dict(showline=True, showticklabels=True),
                          xaxis_title='Date',
                          yaxis_title='Case and Mobility Flow')

        fn = f"case_&_flow_{sra}.png"
        fp = os.path.join(out_dir, fn)
        fig.write_image(fp)

    # fig.show()


# =========================================
#             DTW DISTANCE
# =========================================
def create_heatmaps_dtw_in_phases(df, out_dir, phase_list, flow_selected):
    sra_name = df['name'].unique()

    fig = make_subplots(rows=5, cols=1, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=("inflow_weekly_avg", "outflow_weekly_avg", "withinflow_weekly_avg",
                                        "netflow_weekly_avg", "total_in_within_weekly_avg"))

    for flow in flow_selected:
        z = []

        for phase in phase_list:
            df_phase = df.loc[phase]
            dtw_list = []

            for sra in sra_name:
                df_phase_sra = df_phase.loc[df_phase['name'] == sra]
                dtw_score = dtw(df_phase_sra["cases_weekly_avg"], df_phase_sra[flow])

                dtw_list.append(round(dtw_score, 1))

            z.append(dtw_list)

        fig.add_trace(px.imshow(z, labels=dict(x="San Diego SRA Names", y=flow, color="DTW Distance"),
                                text_auto=True,
                                x=sra_name,
                                # y=['05/01/20-06/24/20', '06/25/20-08/18/20', '08/19/20-10/31/20', '11/01/20-01/23/21',
                                #    '01/24/21-04/03/21']).data[0], row=flow_selected.index(flow) + 1, col=1)  # fix
                                y=['04/01/20-05/31/20', '06/01/20-07/20/20', '07/21/20-08/31/20', '09/01/20-11/05/20',
                                   '11/06/20-01/05/21', '01/06/21-03/27/21']).data[0],
                      row=flow_selected.index(flow) + 1, col=1)
        fig.update_xaxes(tickangle=37, title_font_family="Arial")

    fig.update_layout(title="DTW DISTANCE BETWEEN COVID_19 CASE WEEKLY AVERAGE & HUMAN MOBILITY WEEKLY AVERAGE \n"
                            "IN 41 SAN DIEGO SRAs OVER TIME")

    filename = "DTW_heatmap.html"
    file_path = os.path.join(out_dir, filename)
    fig.write_html(file=file_path, include_plotlyjs="cdn")

    # Save file and show plot
    fn = f"DTW_heatmap.png"    # file name
    fp = os.path.join(out_dir, fn)
    fig.write_image(fp)

    # fig.show()


def create_table_dtw_in_phases(df, out_dir, phase, phase_list, flow_selected):
    """ DTW values -> row -> all rows -> dataframe for each phase -> list of dataframes for all phases """

    sra_name = df['name'].unique()
    df_dtw_tbl_list = []

    for i in range(len(phase_list)):
        df_phase = df.loc[phase_list[i]]
        all_rows = []

        for sra in sra_name:
            df_phase_sra = df_phase.loc[df_phase["name"] == sra]
            each_row = [sra]

            for flow in flow_selected:
                dtw_score = dtw(df_phase_sra["cases_weekly_avg"], df_phase_sra[flow])
                each_row.append(round(dtw_score, 1))

            all_rows.append(each_row)

        df_dtw_tbl = pandas.DataFrame(all_rows)
        df_dtw_tbl.columns = ["SRA", "DTW - Inflow", "DTW - Outflow", "DTW - CWithin", "DTW - Netflow",
                              "DTW - InWithinflow"]
        df_dtw_tbl_list.append(df_dtw_tbl)

        dtw_tbl_html = df_dtw_tbl.sort_values(by=['SRA']).to_html(escape=False, index=False, justify="center")
        text_file = open(f"{out_dir}/DTW_table_{phase[i]}.html", "w")  # edit
        text_file.write(dtw_tbl_html)
        text_file.close()

        df_dtw_tbl.sort_values(by=['SRA']).to_csv(os.path.join(out_dir, f"dtw_tbl_{phase[i]}.csv"))

    return df_dtw_tbl_list


def create_choropleth_map_dtw_in_phases(df_dtw_tbl_list, out_dir, phase):
    sra_shapefile = geopandas.read_file("./data/sra2000/sra2000.shp")
    sra_shapefile['NAME'] = sra_shapefile['NAME'].str.upper()

    # Get coords for geometry centroids
    sra_shapefile['geom_centroid'] = sra_shapefile.centroid
    sra_shapefile['coords'] = sra_shapefile['geom_centroid'].apply(lambda x: x.representative_point().coords[:])
    sra_shapefile['coords'] = [coords[0] for coords in sra_shapefile['coords']]

    # print(type(sra_shapefile))
    # sra_shapefile.plot()
    # plt.show()

    # iterate the phases
    for i in range(len(df_dtw_tbl_list)):
        df_merged = sra_shapefile.merge(df_dtw_tbl_list[i], how='left', left_on=['NAME'], right_on=['SRA'])
        df_merged = df_merged.drop(df_merged.columns[3], axis=1)  # drop column SRA_y (column index 3)

        gdf_dtw = geopandas.GeoDataFrame(df_merged, geometry=df_merged["geometry"], crs='EPSG:4326')

        dtw_cols = df_dtw_tbl_list[i].columns.values.tolist()  # column names: SRA, dtw_in, dtw_outflow...

        for j in range(1, len(dtw_cols)):
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'aspect': 'equal'})
            gdf_dtw.plot(column=dtw_cols[j], legend=True, ax=ax, cmap="Blues_r")
            # range_color=(0, 13), cmap='Blues_r', cmap="winter"

            # Label SRA names in the choropleth maps
            for ind, row in gdf_dtw.iterrows():
                plt.annotate(text=row["NAME"], xy=row["coords"], horizontalalignment="center", fontsize=5)

            # https://geopandas.org/en/stable/docs/user_guide/mapping.html
            plt.title(f"DTW Values ({phase[i]}, {dtw_cols[j]}) in 41 SRAs")

            # Save file
            fn = f"DTW_choropleth_{phase[i]}_{dtw_cols[j]}.png"
            fp = os.path.join(out_dir, fn)
            plt.savefig(fp, bbox_inches="tight")

            # plt.show()


# =========================================
#             GET THE SLOPE
# =========================================


# =================================================================
#        PLOT ONE CLUSTER ONE PLOT (TIME SERIES KMEANS CLUSTERS)
# =================================================================
def plot_1_cluster_1plot(df, out_dir1, out_dir2, phase, phase_list, phase_ranges, metrics):
    date_format = "%Y-%m-%d"

    for i in range(len(phase_list)):
        df_phase = df.loc[phase_list[i]]
        df_grouped = df_phase.groupby('name')
        forty_one_sra_names = df_grouped.groups.keys()

        for col in metrics:  # loop over `metric name list`

            for k in range(3, 15):  # change here if change k
                # https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook

                forty_one_ts = collect_41_ts(col, df_grouped, forty_one_sra_names)
                results, square_side = cluster_41_ts_into_k_groups(forty_one_ts, k)
                viz_k_clusters(col, date_format, forty_one_ts, i, k, phase, phase_ranges, results, square_side)
                save_to_file(col, i, k, out_dir1, out_dir2, phase, results)


def collect_41_ts(col, df_grouped, forty_one_sra_names):
    """
    :param col: each metric in metric name list
    :param df_grouped: df_phase grouped by sra names
    :param forty_one_sra_names: empty list in the beginning
    :return: forty_one_ts_dfs is a list of 41 dfs (representing 41 case_weekly_avg time series of 41 SRAs)
    """
    forty_one_ts = []

    for sra in forty_one_sra_names:  # loop over 41 `sra name list`
        each_ts = get_each_ts(df_grouped, sra, col)  # df = date, cases_weekly_avg
        forty_one_ts.append(each_ts)

    # print(f"There are {len(forty_one_ts_dfs)} time series")
    # series_lengths = {len(series) for series in forty_one_ts_dfs}
    # print("Series length: ", series_lengths)
    # nan_counter(forty_one_ts_dfs)  # no NAN

    return forty_one_ts


def get_each_ts(df_grouped, sra_name, col):
    sra_df = df_grouped.get_group(sra_name)     # dataframe for each sra
    date_indx = sra_df.columns.get_loc("date")  # get the row index of the "date" column
    col_avg_indx = sra_df.columns.get_loc(col)  # get the row index of each column in metrics

    each_ts = sra_df.iloc[:, [date_indx, col_avg_indx]]
    each_ts.set_index("date", inplace=True)
    each_ts.sort_index(inplace=True)

    return each_ts


def cluster_41_ts_into_k_groups(forty_one_ts_dfs, k):
    """ TIMESERIESKMEANS """

    # cluster_count = math.ceil(math.sqrt(len(forty_one_ts_dfs)))  # square root of 41 (=6?)
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw")
    results = model.fit_predict(forty_one_ts_dfs)
    square_side = math.ceil(math.sqrt(math.sqrt(len(forty_one_ts_dfs))))
    # math. ceil() method rounds a number UP to the nearest integer, if necessary, and returns the result.
    return results, square_side


def viz_k_clusters(col, date_format, forty_one_ts_dfs, i, k, phase, phase_ranges, results, square_side):
    """VISUALIZE KMEAN CLUSTERS - each cluster in each plot, the big plot has k minor plots for k clusters
       DTW BARYCENTER AVERAGING (DBA)
    """

    plot_count = math.ceil(math.sqrt(k))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle(f'{k} CLUSTERS - {col} - {phase[i]}')

    row = 0
    column = 0

    for cluster_label in set(results):
        each_cluster = []

        for index in range(len(results)):  # loop through each of the 41 time series
            if results[index] == cluster_label:  # if time series group index = cluster number
                axs[row, column].plot(forty_one_ts_dfs[index], c="gray", alpha=0.4)
                each_cluster.append(forty_one_ts_dfs[index])

        plot_dba(axs, column, date_format, each_cluster, i, phase_ranges, row)  # draw DBA line

        column, row = set_accessories_for_cluster_plots(axs, column, date_format, i, k, phase_ranges, plot_count, row,
                                                        square_side)


def set_accessories_for_cluster_plots(axs, column, date_format, i, k, phase_ranges, plot_count, row, square_side):
    """SET TITLE, X-LIM, Y-LIM FOR CLUSTER PLOTS"""

    for num in range(1, k + 1):
        axs[row, column].set_title("Cluster " + str(row * square_side + column))  # name each plot

    # set xlim to avoid 1970 - 2020 in plot
    axs[row, column].set_xlim([datetime.date(datetime.strptime(phase_ranges[i][0], date_format)),
                               datetime.date(datetime.strptime(phase_ranges[i][1], date_format))])

    # set ylim from 0 to 1
    axs[row, column].set_ylim(0.0, 1.0)

    # subplots go from left to right, high to low order
    column += 1

    if column % plot_count == 0:
        row += 1
        column = 0

    return column, row


def save_to_file(col, i, k, out_dir1, out_dir2, phase, results):
    # Clusters
    fn = f"1_cluster_1plot_{col}_{phase[i]}_k{k}.png"
    fp = os.path.join(out_dir1, fn)
    plt.savefig(fp, bbox_inches="tight")

    # - UNCOMMENT TO RUN
    # Cluster distribution
    cluster_c = [len(results[results == i]) for i in range(k)]
    cluster_n = ["Cluster " + str(i) for i in range(k)]
    plt.figure(figsize=(15, 5))
    plt.title(f"Cluster Distribution for KMeans - {col} - {phase[i]} - Cluster number = {k}")
    plt.bar(cluster_n, cluster_c)
    fn2 = f"Cluster_distribution_{col}_{phase[i]}_k{k}.png"
    fp2 = os.path.join(out_dir2, fn2)
    plt.savefig(fp2, bbox_inches="tight")

    # plt.show()


def plot_dba(axs, column, date_format, each_cluster, i, phase_ranges, row):
    """DTW BARYCENTER AVERAGING (DBA)"""

    if len(each_cluster) > 0:
        end_date = datetime.strptime(phase_ranges[i][1], date_format)
        begin_date = datetime.strptime(phase_ranges[i][0], date_format)
        date_index = pandas.date_range(begin_date, end_date, freq='D')

        # calculate BDA and put it in a dataframe using date_index as index
        dba_nv_cluster = numpy.ravel(dtw_barycenter_averaging(each_cluster, max_iter=50, tol=1e-3))
        dba_nv_cluster_df = pandas.DataFrame(dba_nv_cluster, index=date_index)

        # plot BDA line in red
        axs[row, column].plot(dba_nv_cluster_df, c="red")


# ===========================================
#           PLOT K CLUSTERS ONE PLOT
# ===========================================
def plot_k_clusters_1plot(df, out_dir, phase, phase_list, phase_ranges, metrics):
    date_format = "%Y-%m-%d"

    for col in metrics:  # loop over 6 metrics

        for i in range(len(phase_list)):  # loop over 5 phases
            df_phase = df.loc[phase_list[i], [col, "name", "date"]]  # df for col and name in each phase
            df_grouped = df_phase.groupby('name')
            fortyone_sra_names = df_grouped.groups.keys()
            fortyone_col_ts_dfs = collect_41_ts(col, df_grouped, fortyone_sra_names)  # collect 41 time series

            # VISUALIZE K CLUSTERS IN ONE PLOT
            r = 0
            c = 0
            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 25))  # change numbers if change k
            fig.suptitle(f'k_clusters_1plot_{col}_{phase[i]}')

            # visualize 41 time series in grey color (without DBA red line) in the last subplot (col = 2, row = 2)
            for j in range(41):
                axs[3, 3].plot(fortyone_col_ts_dfs[j], c="gray", alpha=0.4)  # change numbers if change k
            axs[3, 3].set_title(f"No K")  # change numbers if change k
            axs[3, 3].set_ylim(0.0, 1.0)  # change numbers if change k

            # loop over k values from 3 to 17
            for k in range(3, 18):  # change numbers if change k
                results, _ = cluster_41_ts_into_k_groups(fortyone_col_ts_dfs, k)
                # plot_count = math.ceil(math.sqrt(k))

                # loop over k labels in the results of k mean clustering (e.g. k = 3 has three labels)
                for cluster_label in set(results):
                    each_cluster = []

                    # plot 41 time series in grey color in each subplot
                    for index in range(len(results)):
                        if results[index] == cluster_label:
                            each_cluster.append(fortyone_col_ts_dfs[index])
                            axs[r, c].plot(fortyone_col_ts_dfs[index], c="gray", alpha=0.4)

                    plot_dba(axs, c, date_format, each_cluster, i, phase_ranges, r)  # add BDA red line

                    # xet title, y-lim
                    axs[r, c].set_title(f"K = {k}")
                    axs[r, c].set_ylim(0.0, 1.0)

                # subplots go from left to right, high to low order
                c += 1
                if c % 4 == 0:  # change numbers if change k
                    r += 1
                    c = 0

            # save to file and show k plots
            fn = f"k_clusters_1plot_{col}_{phase[i]}.png"
            fp = os.path.join(out_dir, fn)
            plt.savefig(fp, bbox_inches="tight")


def create_random_best_k_tbl(cols_selected):
    """ use this table when we do not have specific k values """

    # Create 2D Numpy array of 5 rows and 6 columns, filled with random values from 5 to 10
    random_data = numpy.random.randint(5, 11, size=(5, 6))
    # Create a Dataframe with random values using 2D numpy Array
    best_k_df = pandas.DataFrame(random_data, columns=cols_selected)
    return best_k_df


def create_best_k_tbl(cols_selected):
    """ the dataframe numbers are the best k values for each of the six metrics in each of five phases"""
    # case = [5, 7, 4, 5, 4]
    # inflow = [17, 13, 16, 16, 14]
    # outflow = [10, 13, 15, 11, 17]
    # withinflow = [10, 17, 16, 16, 13]
    # netflow = [13, 14, 13, 15, 14]
    # in_within = [16, 17, 17, 16, 15]

    # cols_selected.insert(0, "phase")

    # phase = [1, 2, 3, 4, 5, 6]
    case = [3, 3, 3, 5, 7, 7]
    inflow = [8, 15, 12, 11, 11, 10]  # example inflow 1st, 2nd phase
    outflow = [8, 10, 11, 12, 11, 9]
    withinflow = [10, 13, 14, 14, 12, 11]
    netflow = [10, 16, 13, 12, 15, 17]
    in_within = [16, 13, 13, 10, 13, 16]

    # zipped = list(zip(phase, case, inflow, outflow, withinflow, netflow, in_within))
    zipped = list(zip(case, inflow, outflow, withinflow, netflow, in_within))
    best_k_df = pandas.DataFrame(zipped, columns=cols_selected)
    # best_k_df_html = best_k_df.to_html(escape=False, index=False, justify="center")
    # text_file = open(f"./output_method1_phases/2_clusters/best_k_table.html", "w")
    # text_file.write(best_k_df_html)
    # text_file.close()

    return best_k_df


# =====================================================
#   GET FINAL RESULTS TABLE (USING BEST K VALUES)
# =====================================================
def build_result_tables(df, phase_list, metrics, out_dir, phase_ranges, phase,
                        out_dir5_1_html, out_dir5_2_html, out_dir5_3_html, out_dir5_4_html):
    """ To get a list of 5 results tables, each having 36 columns and 41 rows, printed out to html """

    results_df_list = []

    for i in range(len(phase_list)):  # (0, 1, 2, 3, 4, 5)
        best_k_df = create_best_k_tbl(metrics)

        df_phase = df.loc[phase_list[i]]
        df_grouped = df_phase.groupby('name')
        forty_one_sra_names = df_grouped.groups.keys()  # (1) first column of the final table: sra

        final_df = pandas.DataFrame.from_dict(forty_one_sra_names)  # column name = 0; need to rename
        final_df.rename(columns={0: 'SRA Names'}, inplace=True)

        list_of_col_label_lists = []
        list_of_col_label_with_outlier_lists = []
        list_of_col_tsnum_lists = []

        dtw_score_list_in, dtw_score_list_out, dtw_score_list_within, dtw_score_list_net, dtw_score_list_in_within \
            = ([] for i in range(5))
        dtw_score_list = [dtw_score_list_in, dtw_score_list_out, dtw_score_list_within, dtw_score_list_net,
                          dtw_score_list_in_within]

        for col in metrics:  # metrics include "cases_weekly_avg"
            forty_one_ts_dfs = []
            col_index = metrics.index(col)
            best_k = best_k_df.iloc[i, col_index]

            for sra in forty_one_sra_names:
                sra_df = df_grouped.get_group(sra)

                for num in range(5):
                    if col == metrics[num + 1]:
                        dtw_score = dtw(sra_df["cases_weekly_avg"], sra_df[col])
                        dtw_score_list[num].append(dtw_score)  # (2) five dtw_score columns of the final table

                # (3) collect six metrics columns for the final table
                ts_df = get_each_ts(df_grouped, sra, col)
                forty_one_ts_dfs.append(ts_df)

            model = TimeSeriesKMeans(n_clusters=best_k, metric="dtw")
            results = model.fit_predict(forty_one_ts_dfs)

            col_label_list, col_label_with_outlier_list, col_tsnum_list = \
                update_41ts_with_labels_and_tsnum(best_k, results, forty_one_ts_dfs, phase_ranges, i)

            list_of_col_label_lists.append(col_label_list)
            list_of_col_label_with_outlier_lists.append(col_label_with_outlier_list)
            list_of_col_tsnum_lists.append(col_tsnum_list)

            # (4) add results to final_df
            final_df[f"Cluster_Num_{col}"] = results.tolist()

        # Add (2) and (3) groups of columns to final_df
        for num in range(5):  # 0, 1, 2, 3, 4
            final_df["DTW_" + metrics[num + 1]] = dtw_score_list[num]

        for num in range(6):  # 0, 1, 2, 3, 4, 5
            final_df["Label_" + metrics[num]] = list_of_col_label_lists[num]
            final_df["Label_Outlier_" + metrics[num]] = list_of_col_label_with_outlier_lists[num]
            final_df["TS_Number_" + metrics[num]] = list_of_col_tsnum_lists[num]

        returns_df = format_result_tables(final_df, i, out_dir, phase, results_df_list)

        categorize_result_tables(results_df=returns_df, i=i,
                                 out_dir5_1_html=out_dir5_1_html,
                                 out_dir5_2_html=out_dir5_2_html,
                                 out_dir5_3_html=out_dir5_3_html,
                                 out_dir5_4_html=out_dir5_4_html,
                                 phase=phase)

    return results_df_list


def update_41ts_with_labels_and_tsnum(best_k, results, forty_one_ts_dfs, phase_ranges, i):
    """
     Update_41ts_with_labels_and_tsnum using the lookup table
    :param best_k: int
    :param results: a list of 41 elements, each having a value from 0 to k-1
    :param forty_one_ts_dfs:
    :param phase_ranges: [['2020-05-01', '2020-6-24'], ['2020-06-25', '2020-8-18'], ['2020-08-19', '2020-10-31'],
                    ['2020-11-01', '2021-1-23'], ['2021-01-24', '2021-4-3']]
    :param i: int
    :return: a list of 41 new labels updated from using the label look up table;
             this is for one metric in one phase.
    """
    lookup_df = create_lookup_tbl(best_k, results, forty_one_ts_dfs, phase_ranges, i)

    convert_label_list = results.tolist()
    convert_label_with_outlier_list = results.tolist()
    convert_ts_num_list = results.tolist()

    # same structure of results but will need replace cluster label with line label

    for k in range(best_k):
        # retrieve all indexes in list for value distinct_cls_label
        indexes = find_indexes(convert_label_list, k)

        for index in indexes:
            convert_label_list[index] = lookup_df.iloc[k]["line_label"]
            convert_ts_num_list[index] = lookup_df.iloc[k]["ts_num"]
            convert_label_with_outlier_list[index] = lookup_df.iloc[k]["line_label_with_outlier"]

    return convert_label_list, convert_label_with_outlier_list, convert_ts_num_list


def create_lookup_tbl(best_k, results, forty_one_ts_dfs, phase_ranges, i):
    """create look-up table of assigned line labels to update 41ts with labels in next step
    two columns - one for k values from 0 to k - 1; one for label of corresponding k value, such as 'stable low'
    lookup table is created for each phase and metric
    """

    lookup_tbl = pandas.DataFrame(list(range(best_k)))
    ts_num_list = []
    label_list = []
    label_list_with_outlier = []

    for distinct_cls_label in range(best_k):
        # each cluster in best_k (int) clusters has certain number of metric time series. One metric time series (= 1
        # col_ts_df having 1 date index and 1 metric col) will have best_k (int) number of ts_clusters after the loop
        ts_per_cl = 0
        ts_in_cl = []

        for result in results:
            result_ind = results.tolist().index(result)  # vs. using `for i, va in enumerate(mylist)`
            if result == distinct_cls_label:
                ts_per_cl += 1
                ts_in_cl.append(forty_one_ts_dfs[result_ind])

        ts_num_list.append(ts_per_cl)
        assigned_label_with_outlier, assigned_label = assign_label(ts_in_cl, phase_ranges, i)

        label_list.append(assigned_label)
        label_list_with_outlier.append(assigned_label_with_outlier)

    lookup_tbl["ts_num"] = ts_num_list
    lookup_tbl["line_label"] = label_list
    lookup_tbl["line_label_with_outlier"] = label_list_with_outlier

    return lookup_tbl


def assign_label(ts_cluster, phase_ranges, i):
    """ calculate slope and intercepts then assign label for the line based on the label definition;
    return assigned_label - string value - such as 'increasing high', 'stable low' """

    date_format = "%Y-%m-%d"

    dba_vals = numpy.ravel(dtw_barycenter_averaging(ts_cluster, max_iter=50, tol=1e-3))
    end_date = datetime.strptime(phase_ranges[i][1], date_format)
    begin_date = datetime.strptime(phase_ranges[i][0], date_format)
    day_num = (end_date - begin_date).days  # convert timedelta to int = 69
    days_array = numpy.array(range(day_num + 1))

    regr_results = scipy.stats.linregress(days_array, dba_vals)
    slope = round(regr_results.slope, 3)
    intercept = regr_results.intercept
    # p_value = round(regr_results.pvalue, 2)

    assigned_label_with_outlier = ""
    assigned_label = define_label(slope, intercept)

    if len(ts_cluster) == 1:
        assigned_label_with_outlier = define_label(slope, intercept) + ", outlier"

    return assigned_label_with_outlier, assigned_label


def define_label(slope, intercept, th_slope=0, th_intercept_low=0.2, th_intercept_up=0.6):
    """ define label for each line based on slope and intercept """

    if slope > th_slope:
        if intercept >= th_intercept_up:
            line_label = "increase high"
        elif th_intercept_low <= intercept < th_intercept_up:
            line_label = "increase mid"
        else:
            line_label = "increase low"

    elif slope < th_slope:
        if intercept >= th_intercept_up:
            line_label = "decrease high"
        elif th_intercept_low <= intercept < th_intercept_up:
            line_label = "decrease mid"
        else:
            line_label = "decrease low"

    else:
        if intercept >= th_intercept_up:
            line_label = "stable high"
        elif th_intercept_low <= intercept < th_intercept_up:
            line_label = "stable mid"
        else:
            line_label = "stable low"

    return line_label


# ==========================
#   FORMAT FINAL RESULTS
# ==========================
def format_result_tables(final_df, i, out_dir, phase, results_df_list):
    """ format five result tables """
    rename_columns(final_df)  # rename columns
    results_df = create_l1_l9_labels(final_df)  # create L1-9 labels
    results_df = rearrange_column_position(results_df)  # rearrange column position

    results_df_list.append(results_df)

    results_df_html = results_df.to_html(escape=False, index=False, justify="center")
    text_file = open(f"{out_dir}/results_appendix_{phase[i]}.html", "w")
    text_file.write(results_df_html)
    text_file.close()

    return results_df


def rename_columns(final_df):
    # Rename and rearrange columns
    final_df.rename(columns={"Label_cases_weekly_avg": "case_Label",
                             "Label_inflow_weekly_avg": "in_Label",
                             "Label_outflow_weekly_avg": "out_Label",
                             "Label_withinflow_weekly_avg": "within_Label",
                             "Label_netflow_weekly_avg": "net_Label",
                             "Label_total_in_within_weekly_avg": "in_within_Label",
                             "Label_Outlier_cases_weekly_avg": "case_Outlier",
                             "Label_Outlier_inflow_weekly_avg": "in_Outlier",
                             "Label_Outlier_outflow_weekly_avg": "out_Outlier",
                             "Label_Outlier_withinflow_weekly_avg": "within_Outlier",
                             "Label_Outlier_netflow_weekly_avg": "net_Outlier",
                             "Label_Outlier_total_in_within_weekly_avg": "in_within_Outlier",
                             "DTW_inflow_weekly_avg": "in_DTW_Value",
                             "DTW_outflow_weekly_avg": "out_DTW_Value",
                             "DTW_withinflow_weekly_avg": "within_DTW_Value",
                             "DTW_netflow_weekly_avg": "net_DTW_Value",
                             "DTW_total_in_within_weekly_avg": "in_within_DTW_Value",
                             "Cluster_Num_cases_weekly_avg": "case_Cluster_Num",
                             "Cluster_Num_inflow_weekly_avg": "in_Cluster_Num",
                             "Cluster_Num_outflow_weekly_avg": "out_Cluster_Num",
                             "Cluster_Num_withinflow_weekly_avg": "within_Cluster_Num",
                             "Cluster_Num_netflow_weekly_avg": "net_Cluster_Num",
                             "Cluster_Num_total_in_within_weekly_avg": "in_within_Cluster_Num",
                             "TS_Number_cases_weekly_avg": "case_TS_Count",
                             "TS_Number_inflow_weekly_avg": "in_TS_Count",
                             "TS_Number_outflow_weekly_avg": "out_TS_Count",
                             "TS_Number_withinflow_weekly_avg": "within_TS_Count",
                             "TS_Number_netflow_weekly_avg": "net_TS_Count",
                             "TS_Number_total_in_within_weekly_avg": "in_within_TS_Count"}, inplace=True)


def create_l1_l9_labels(final_df):
    result_df = final_df.copy(deep=True)

    result_df["case_abbre"] = result_df["case_Label"]
    result_df["in_abbre"] = result_df["in_Label"]
    result_df["out_abbre"] = result_df["out_Label"]
    result_df["within_abbre"] = result_df["within_Label"]
    result_df["net_abbre"] = result_df["net_Label"]
    result_df["in_within_abbre"] = result_df["in_within_Label"]

    for column in result_df[["case_abbre", "in_abbre", "out_abbre", "within_abbre", "net_abbre", "in_within_abbre"]]:
        column_ind = result_df.columns.get_loc(column)  # 12 to 17
        columnSeriesObj = result_df[column]  # (0, 'stable low') (1, 'stable low') etc.
        col_content = columnSeriesObj.values.tolist()
        # turn ndarray to list (41 elements): ['stable low', 'stable low', etc.]

        for index, val in enumerate(col_content):
            if val == "increase high":
                result_df.iloc[index, column_ind] = "IH"
            elif val == "increase mid":
                result_df.iloc[index, column_ind] = "IM"
            elif val == "increase low":
                result_df.iloc[index, column_ind] = "IL"
            elif val == "decrease high":
                result_df.iloc[index, column_ind] = "DH"
            elif val == "decrease mid":
                result_df.iloc[index, column_ind] = "DM"
            elif val == "decrease low":
                result_df.iloc[index, column_ind] = "DL"
            elif val == "stable high":
                result_df.iloc[index, column_ind] = "SH"
            elif val == "stable mid":
                result_df.iloc[index, column_ind] = "SM"
            elif val == "stable low":
                result_df.iloc[index, column_ind] = "SL"

    return result_df


def rearrange_column_position(result_df):
    result_df = result_df[["SRA Names",
                           "case_Cluster_Num", "case_TS_Count", "case_Outlier", "case_Label", "case_abbre",
                           "in_Cluster_Num", "in_TS_Count", "in_Outlier", "in_Label", "in_abbre", "in_DTW_Value",
                           "out_Cluster_Num", "out_TS_Count", "out_Outlier", "out_Label", "out_abbre", "out_DTW_Value",
                           "within_Cluster_Num", "within_TS_Count", "within_Outlier", "within_Label", "within_abbre",
                           "within_DTW_Value",
                           "net_Cluster_Num", "net_TS_Count", "net_Outlier", "net_Label", "net_abbre", "net_DTW_Value",
                           "in_within_Cluster_Num", "in_within_TS_Count", "in_within_Outlier", "in_within_Label",
                           "in_within_abbre", "in_within_DTW_Value"]]

    return result_df


# =================================
#   GET INSIGHT FROM FINAL RESULTS
# =================================
def categorize_result_tables(results_df, i, out_dir5_1_html, out_dir5_2_html, out_dir5_3_html, out_dir5_4_html,
                             phase):
    """collect increase/stable/decrease/outlier dataframes"""

    get_increase_df(results_df, out_dir5_1_html, phase, i)
    get_stable_df(results_df, out_dir5_2_html, phase, i)
    get_decrease_df(results_df, out_dir5_3_html, phase, i)
    get_outlier_df(results_df, out_dir5_4_html, phase, i)


def get_increase_df(result_df, out_dir1, phase, i):
    """ return df that has increasing case and increasing flow
        flow = string, such as 'in_abbre'.
        out_dir1 = out_dir5_1_html
        only print out df if df has content (row number > 0)"""

    flow_list = ["in_abbre", "out_abbre", "within_abbre", "net_abbre", "in_within_abbre"]

    for flow in flow_list:
        increase_ll_df = result_df[(result_df["case_abbre"] == "IL") & (result_df[flow] == "IL")]
        increase_lm_df = result_df[(result_df["case_abbre"] == "IL") & (result_df[flow] == "IM")]
        increase_lh_df = result_df[(result_df["case_abbre"] == "IL") & (result_df[flow] == "IH")]
        increase_ml_df = result_df[(result_df["case_abbre"] == "IM") & (result_df[flow] == "IL")]
        increase_mm_df = result_df[(result_df["case_abbre"] == "IM") & (result_df[flow] == "IM")]
        increase_mh_df = result_df[(result_df["case_abbre"] == "IM") & (result_df[flow] == "IH")]
        increase_hl_df = result_df[(result_df["case_abbre"] == "IH") & (result_df[flow] == "IL")]
        increase_hm_df = result_df[(result_df["case_abbre"] == "IH") & (result_df[flow] == "IM")]
        increase_hh_df = result_df[(result_df["case_abbre"] == "IH") & (result_df[flow] == "IH")]

        increase_df = pandas.concat([increase_ll_df, increase_lm_df, increase_lh_df, increase_ml_df, increase_mm_df,
                                     increase_mh_df, increase_hh_df, increase_hm_df, increase_hl_df])

        row_num = increase_df.shape[0]
        # row_num = len(increase_df.index)

        if row_num > 0:  # save to html only when df has content; in fact some dfs are empty
            increase_df_html = increase_df.to_html(escape=False, index=False, justify="center")
            text_file = open(f"{out_dir1}/increase_{phase[i]}_{flow}.html", "w")
            text_file.write(increase_df_html)
            text_file.close()


def get_stable_df(result_df, out_dir1, phase, i):
    """ return df that has stable case and stable inflow
        flow = string, such as 'in_abbre'.
        out_dir1 = out_dir5_2_html
        only print out df if df has content (row number > 0)"""

    flow_list = ["in_abbre", "out_abbre", "within_abbre", "net_abbre", "in_within_abbre"]

    for flow in flow_list:
        stable_ll_df = result_df[(result_df["case_abbre"] == "SL") & (result_df[flow] == "SL")]
        stable_lm_df = result_df[(result_df["case_abbre"] == "SL") & (result_df[flow] == "SM")]
        stable_lh_df = result_df[(result_df["case_abbre"] == "SL") & (result_df[flow] == "SH")]
        stable_ml_df = result_df[(result_df["case_abbre"] == "SM") & (result_df[flow] == "SL")]
        stable_mm_df = result_df[(result_df["case_abbre"] == "SM") & (result_df[flow] == "SM")]
        stable_mh_df = result_df[(result_df["case_abbre"] == "SM") & (result_df[flow] == "SH")]
        stable_hl_df = result_df[(result_df["case_abbre"] == "SH") & (result_df[flow] == "SL")]
        stable_hm_df = result_df[(result_df["case_abbre"] == "SH") & (result_df[flow] == "SM")]
        stable_hh_df = result_df[(result_df["case_abbre"] == "SH") & (result_df[flow] == "SH")]
        stable_df = pandas.concat([stable_ll_df, stable_lm_df, stable_lh_df, stable_ml_df, stable_mm_df, stable_mh_df,
                                   stable_hh_df, stable_hm_df, stable_hl_df])
        row_num = stable_df.shape[0]

        if row_num > 0:  # save to html only when df has content; in fact some dfs are empty
            stable_df_html = stable_df.to_html(escape=False, index=False, justify="center")
            text_file = open(f"{out_dir1}/stable_{phase[i]}_{flow}.html", "w")
            text_file.write(stable_df_html)
            text_file.close()


def get_decrease_df(result_df, out_dir1, phase, i):
    """ return df that has decreasing case and decreasing inflow
        flow = string, such as 'in_abbre'.
        out_dir1 = out_dir5_3_html
        only print out df if df has content (row number > 0)"""

    flow_list = ["in_abbre", "out_abbre", "within_abbre", "net_abbre", "in_within_abbre"]

    for flow in flow_list:
        decrease_ll_df = result_df[(result_df["case_abbre"] == "DL") & (result_df[flow] == "DL")]
        decrease_lm_df = result_df[(result_df["case_abbre"] == "DL") & (result_df[flow] == "DM")]
        decrease_lh_df = result_df[(result_df["case_abbre"] == "DL") & (result_df[flow] == "DH")]
        decrease_ml_df = result_df[(result_df["case_abbre"] == "DM") & (result_df[flow] == "DL")]
        decrease_mm_df = result_df[(result_df["case_abbre"] == "DM") & (result_df[flow] == "DM")]
        decrease_mh_df = result_df[(result_df["case_abbre"] == "DM") & (result_df[flow] == "DH")]
        decrease_hl_df = result_df[(result_df["case_abbre"] == "DH") & (result_df[flow] == "LD")]
        decrease_hm_df = result_df[(result_df["case_abbre"] == "DH") & (result_df[flow] == "DM")]
        decrease_hh_df = result_df[(result_df["case_abbre"] == "DH") & (result_df[flow] == "DH")]
        decrease_df = pandas.concat([decrease_ll_df, decrease_lm_df, decrease_lh_df, decrease_ml_df, decrease_mm_df,
                                     decrease_mh_df, decrease_hh_df, decrease_hm_df, decrease_hl_df])
        row_num = decrease_df.shape[0]

        if row_num > 0:  # save to html only when df has content; in fact some dfs are empty
            decrease_df_html = decrease_df.to_html(escape=False, index=False, justify="center")
            text_file = open(f"{out_dir1}/decrease_{phase[i]}_{flow}.html", "w")
            text_file.write(decrease_df_html)
            text_file.close()


def get_outlier_df(result_df, out_dir1, phase, i):
    """ return df that has outliers
        flow = string, such as 'in_abbre'.
        out_dir1 = out_dir5_4_html
        only print out df if df has content (row number > 0)"""

    flow_list = ["case_abbre", "in_abbre", "out_abbre", "within_abbre", "net_abbre", "in_within_abbre"]

    for flow in flow_list:
        outlier_df = result_df[result_df[flow].str.contains("outlier")]
        row_num = outlier_df.shape[0]

        if row_num > 0:  # save to html only when df has content; in fact some dfs are empty
            outlier_df_html = outlier_df.to_html(escape=False, index=False, justify="center")
            text_file = open(f"{out_dir1}/outlier_{phase[i]}_{flow}.html", "w")
            text_file.write(outlier_df_html)
            text_file.close()


def draw_conclusion(sra_names, results_df_list, phase_list, out_dir):
    """ results_df_list = list of results_df in each phase;
        phase_list = divide_time(df_data_minmax)
    """

    flow_abbre = ["in_Label", "out_Label", "within_Label", "net_Label", "in_within_Label"]
    flow_metrics = ["inflow", "outflow", "withinflow", "netflow", "total_in_within"]
    insight_df_list = []

    # 1. Iterate flows
    for flw_idx, flow in enumerate(flow_metrics):
        insight_df = pandas.DataFrame({"SRA Names": sra_names})

        # 2. Iterate phases
        for i in range(len(phase_list)):
            phase_df = results_df_list[i]
            insight_df[f"phase_{i + 1}"] = ""

            ac_idx = phase_df.columns.get_loc("case_Label")
            af_idx = phase_df.columns.get_loc(flow_abbre[flw_idx])
            ph_idx = insight_df.columns.get_loc(f"phase_{i + 1}")

            # 3. Iterate SRAs
            for row_idx, row in insight_df.iterrows():
                ac_val = phase_df.iloc[row_idx, ac_idx]
                af_val = phase_df.iloc[row_idx, af_idx]

                if "stable" in ac_val and "stable" in af_val:
                    insight_df.iloc[row_idx, ph_idx] = "stable"
                elif "increase" in ac_val and "increase" in af_val:
                    insight_df.iloc[row_idx, ph_idx] = "increase"
                elif "decrease" in ac_val and "decrease" in af_val:
                    insight_df.iloc[row_idx, ph_idx] = "decrease"
                else:
                    insight_df.iloc[row_idx, ph_idx] = "-"

        insight_df.to_csv(f"{out_dir}/insight_df_{flow}.csv")

        insight_df_html = insight_df.to_html(escape=False, index=False, justify="center")
        text_file = open(f"{out_dir}/insight_df_{flow}.html", "w")
        text_file.write(insight_df_html)
        text_file.close()

        insight_df_list.append(insight_df)

    return insight_df_list
