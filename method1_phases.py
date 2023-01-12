# first method: thesis_pham_phases
# folder viz_oct_data_overall is final for visualization (11/22/2022)
# folder output_method1_phases is final for doing the first method thesis_pham_phases (11/22/2022)
import os
import sys
import pandas
import utils_pham


def main():
    code_run_start, start_time = utils_pham.start_run_code()
    run_phases()
    # there are two flags in this code
    utils_pham.print_code_run_time(code_run_start, start_time)


def run_phases():
    case_n_flow_cols, flow_cols, flow_cols_short, cols_to_viz, phase_dates, phase_names = \
        utils_pham.define_flow_and_case_cols_and_phases()

    out_dir1_1, out_dir1_2, out_dir1_3, out_dir2_1, out_dir3_1, out_dir3_2, out_dir3_3, out_dir4_1, \
        out_dir5_1_html, out_dir5_2_html, out_dir5_3_html, out_dir5_4_html, out_dir5_5_html, outdir_14days \
        = utils_pham.create_output_folders()

    df_data, sra_names = import_and_process_data()

    # utils_pham.test_normal_distribution(df_data, case_n_flow_cols)  # did not work

    df_data_minmax = utils_pham.normalize_data(df_data)
    phase_list = utils_pham.divide_time(df_data_minmax)
    cal_netflow_and_7days_rolling_average(df_data_minmax)

    # Visualize data
    flag_viz_data = False
    if flag_viz_data:
        visualize_all_data(df_data_minmax, cols_to_viz, out_dir1_1, out_dir1_2, out_dir1_3)
        visualize_phase_data(cols_to_viz, df_data_minmax, phase_names, phase_list)
        # NOTE
        # from beginning to here took about an hour, or an hour and 15 minutes
        # then from here to end took 20 minutes. Total is an hour and a half.

    # Visualize DTW values
    flag_cal_and_visualize_dtw = False
    if flag_cal_and_visualize_dtw:
        cal_and_visualize_dtw(df_data_minmax, flow_cols, phase_list, out_dir2_1, phase_names)

    # Visualize time series clusters
    flag_plot_ts_clusters = False
    if flag_plot_ts_clusters:
        plot_ts_clusters(cols_selected=case_n_flow_cols,
                         df_data_minmax=df_data_minmax,
                         out_dir3_1=out_dir3_1,
                         out_dir3_2=out_dir3_2,
                         out_dir3_3=out_dir3_3,
                         phase=phase_names,
                         phase_list=phase_list,
                         phase_ranges=phase_dates)
        # this call completely independent from calculating dtw step above

    # Get results for final conclusions
    build_result_tables_and_draw_conclusions(cols_selected=case_n_flow_cols,
                                             df_data_minmax=df_data_minmax,
                                             out_dir4_1=out_dir4_1,
                                             out_dir5_1_html=out_dir5_1_html,
                                             out_dir5_2_html=out_dir5_2_html,
                                             out_dir5_3_html=out_dir5_3_html,
                                             out_dir5_4_html=out_dir5_4_html,
                                             out_dir5_5_html=out_dir5_5_html,
                                             phase=phase_names,
                                             phase_list=phase_list,
                                             phase_ranges=phase_dates,
                                             sra_names=sra_names)


def build_result_tables_and_draw_conclusions(cols_selected, df_data_minmax, out_dir4_1, out_dir5_1_html,
                                             out_dir5_2_html, out_dir5_3_html, out_dir5_4_html, out_dir5_5_html,
                                             phase, phase_list, phase_ranges, sra_names):
    utils_pham.print_heading("5. DETERMINE BEST K AND GET FINAL RESULTS TABLE ")

    results_df_list = utils_pham.build_result_tables(df=df_data_minmax,
                                                     phase_list=phase_list,
                                                     metrics=cols_selected,
                                                     out_dir=out_dir4_1,
                                                     phase_ranges=phase_ranges,
                                                     phase=phase,
                                                     out_dir5_1_html=out_dir5_1_html,
                                                     out_dir5_2_html=out_dir5_2_html,
                                                     out_dir5_3_html=out_dir5_3_html,
                                                     out_dir5_4_html=out_dir5_4_html)

    utils_pham.draw_conclusion(results_df_list=results_df_list,
                               phase_list=phase_list,
                               out_dir=out_dir5_5_html,
                               sra_names=sra_names)


def plot_ts_clusters(cols_selected, df_data_minmax, out_dir3_1, out_dir3_2, out_dir3_3, phase, phase_list,
                     phase_ranges):
    utils_pham.print_heading("4. TIME SERIES CLUSTERING")

    utils_pham.plot_1_cluster_1plot(df=df_data_minmax,
                                    out_dir1=out_dir3_1,
                                    out_dir2=out_dir3_2,
                                    phase=phase,
                                    phase_list=phase_list,
                                    phase_ranges=phase_ranges,
                                    metrics=cols_selected)

    utils_pham.plot_k_clusters_1plot(df=df_data_minmax,
                                     out_dir=out_dir3_3,
                                     phase=phase,
                                     phase_list=phase_list,
                                     phase_ranges=phase_ranges,
                                     metrics=cols_selected)


def cal_and_visualize_dtw(df_data_minmax, flow_selected, phase_list, out_dir2_1, phase):
    """Create heatmap, geopandas dataframes, and tables of dtw index values"""
    utils_pham.print_heading("3. DTW DISTANCE")

    utils_pham.print_subheading("\n3.1 Calculate DTW distance")
    utils_pham.create_heatmaps_dtw_in_phases(df_data_minmax, out_dir2_1, phase_list, flow_selected)

    utils_pham.print_subheading("\n3.2 Create geopandas dataframes and tables of DTW index values")
    df_dtw_tbl_list = utils_pham.create_table_dtw_in_phases(df_data_minmax, out_dir2_1, phase, phase_list,
                                                            flow_selected)
    utils_pham.create_choropleth_map_dtw_in_phases(df_dtw_tbl_list, out_dir2_1, phase)


def visualize_phase_data(cols_to_viz, df_data_minmax, phase, phase_list):
    """Visualize df_data_minmax in different phases"""
    utils_pham.print_subheading("\n2.3 Visualize phase data")

    for i in range(len(phase_list)):
        df_phase = df_data_minmax.loc[phase_list[i]]

        out_dir_p_1 = "./viz_oct_data_eachphase/df_min_max/" + phase[i] + "/one_metric_one_sra_v2"
        out_dir_p_2 = "./viz_oct_data_eachphase/df_min_max/" + phase[i] + "/one_metric_all_sra_v2"
        out_dir_p_3 = "./viz_oct_data_eachphase/df_min_max/" + phase[i] + "/multiple_metric_one_sra_v2"

        for folder in [out_dir_p_1, out_dir_p_2, out_dir_p_3]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        utils_pham.plot_one_metric_each_sra(df_phase, cols_to_viz, out_dir_p_1)
        utils_pham.plot_one_metric_all_sra(df_phase, cols_to_viz, out_dir_p_2)
        utils_pham.plot_multiple_metric_one_sra(df_phase, out_dir_p_3)


def visualize_all_data(df_data_minmax, cols_to_viz, out_dir1_1, out_dir1_2, out_dir1_3):
    """Visualize df_data_minmax for all data """
    utils_pham.print_subheading("\n2.2 Visualize all data")

    utils_pham.plot_one_metric_each_sra(df_data_minmax, cols_to_viz, out_dir1_1)
    utils_pham.plot_one_metric_all_sra(df_data_minmax, cols_to_viz, out_dir1_2)
    utils_pham.plot_multiple_metric_one_sra(df_data_minmax, out_dir1_3)


def cal_netflow_and_7days_rolling_average(df_data_minmax):
    # 1. Calculate netflows
    df_data_minmax['netflow'] = df_data_minmax['inflow'] - df_data_minmax['outflow']
    df_data_minmax['netflow_sd'] = df_data_minmax['in_sd'] - df_data_minmax['out_sd']
    df_data_minmax['netflow_non_sd'] = df_data_minmax['in_non_sd'] - df_data_minmax['out_non_sd']

    # 2. Calculate 7 day rolling average
    utils_pham.print_subheading("\n1.5. Calculate 7 day rolling average")
    # when calculating 7 day rolling average for df_case: if min_obs != 1, will get nan values
    df_data_minmax['cases_weekly_avg'] = df_data_minmax.groupby(['name'])['new_case'].transform(utils_pham.cal_roll_avg,
                                                                                                num_day=7, min_obs=1)
    flow_cols = ["inflow", "in_sd", "in_non_sd",
                 "outflow", "out_sd", "out_non_sd",
                 "netflow", "netflow_sd", "netflow_non_sd",
                 "withinflow", "total_in_within"]

    for col in flow_cols:
        df_data_minmax[f"{col}_weekly_avg"] = df_data_minmax.groupby(['name'])[col].transform(utils_pham.cal_roll_avg,
                                                                                              num_day=7, min_obs=1)

    # for col in df_data_minmax.columns.values.tolist():
    #     utils_pham.check_negative_values(df_data_minmax, col, "df_data_minmax")


def import_and_process_data():
    utils_pham.print_heading("1. IMPORT DATA")

    df_flow = import_human_mobility_data()

    df_case, sra_names = import_process_covid_data()

    utils_pham.print_subheading("\n1.3 Merge Mobility data and Covid data")

    df_data = df_case.merge(df_flow, how='left', left_on=['sra', 'date'], right_on=['sra_id', 'date'])
    df_data = df_data.drop(['sra_id', 'sra_name'], axis=1)
    # df_data = df_data.sort_values(by=['name', 'date'])
    # cols_all = df_data.columns.values.tolist()

    return df_data, sra_names


def import_process_covid_data():
    utils_pham.print_subheading("\n1.2 COVID-19 data")

    df_case, _, _ = import_covid_data()

    df_case, sra_names = process_covid_data(df_case)

    return df_case, sra_names


def process_covid_data(df_case):
    df_case['new_case'] = df_case['new_case'].fillna(0)  # Convert nan to 0 (41 values replaced)
    df_case['new_case'] = df_case['new_case'].clip(lower=0)  # Convert negative new_case to 0 (91 values replaced)
    # df_case[df_case['new_case'] < 0] = 0 (another way)

    new_case_df, sra_names = fill_up_missing_days_with_values_of_previous_days(df_case)

    # Replace old df_case with new_case_df and fill in 123 missing values (= 41 SRA * 3 days)
    df_case = new_case_df
    utils_pham.check_missing_rows(df_case, "new_case", "the updated df_case (after data processing)")
    utils_pham.check_negative_values(df_case, "new_case", "df_case (after data processing)")

    return df_case, sra_names


def fill_up_missing_days_with_values_of_previous_days(df_case):
    # Fill up three missing days with values of the previous day
    new_case_df = pandas.DataFrame(columns=["sra", "name", "date", "case_acum", "new_case"])
    sra_names = df_case['name'].unique()

    for sra_name in sra_names:
        sra_df = df_case.loc[df_case['name'] == sra_name]

        new_sra = sra_df.loc[sra_df["name"] == sra_name, "sra"].iloc[0]
        case_acum_1 = sra_df.loc[sra_df["date"] == "2021-02-13 00:00:00", "case_acum"].iloc[0]
        case_acum_2 = sra_df.loc[sra_df["date"] == "2021-03-29 00:00:00", "case_acum"].iloc[0]
        case_acum_3 = sra_df.loc[sra_df["date"] == "2021-01-16 00:00:00", "case_acum"].iloc[0]
        new_case_1 = sra_df.loc[sra_df["date"] == "2021-02-13 00:00:00", "new_case"].iloc[0]
        new_case_2 = sra_df.loc[sra_df["date"] == "2021-03-29 00:00:00", "new_case"].iloc[0]
        new_case_3 = sra_df.loc[sra_df["date"] == "2021-01-16 00:00:00", "new_case"].iloc[0]

        dict_to_add = {"sra": [new_sra, new_sra, new_sra],
                       "name": [sra_name, sra_name, sra_name],
                       "date": ["2021-02-14 00:00:00", "2021-03-30 00:00:00", "2021-01-17 00:00:00"],
                       "case_acum": [case_acum_1, case_acum_2, case_acum_3],
                       "new_case": [new_case_1, new_case_2, new_case_3]}

        df_to_add = pandas.DataFrame(dict_to_add)

        sra_df = pandas.concat([sra_df, df_to_add], ignore_index=True)
        sra_df["date"] = pandas.to_datetime(sra_df["date"])  # turn date column from str to date type
        sra_df = sra_df.sort_values(by=["date"])  # then sort date by value
        sra_df.reset_index(inplace=True)  # reset index to include new values
        sra_df = sra_df.drop(['index'], axis=1)  # delete index column created from running previous command

        new_case_df = pandas.concat([new_case_df, sra_df], ignore_index=True)

    new_case_df.reset_index(inplace=True)
    new_case_df = new_case_df.drop(['index'], axis=1)

    return new_case_df, sra_names


def import_covid_data():
    # df_case = pandas.read_csv("./data/sra_case_daily.csv", parse_dates=['date'])        # old data
    df_case = pandas.read_csv("./data/sra_case_daily_aag2022.csv", parse_dates=['date'])  # new data
    utils_pham.check_negative_values(df_case, "new_case", "df_case")
    utils_pham.check_missing_rows(df_case, "new_case", "df_case")  # missing values on 03/30/2020 (new data)
    print("\nFirst day of Covid-19 data:", min(df_case["date"]), "Last day of Covid-19 data:", max(df_case["date"]))
    print("\n41 SRA numbers:", df_case['sra'].unique())
    print("\n41 SRA names:", df_case['name'].unique())

    sra_num = df_case['sra'].unique()
    sra_name = df_case['name'].unique()

    # create a df for sra_name and sra_num
    df_sra_info = pandas.DataFrame([sra_num, sra_name]).transpose()
    df_sra_info.rename(columns={0: "sra_num", 1: "sra_name"}, inplace=True)

    return df_case, sra_num, sra_name, df_sra_info


def import_human_mobility_data():
    utils_pham.print_subheading("\n1.1 Human Mobility data")

    # df_flow = pandas.read_csv("./data/sd_flow_sra.csv", parse_dates=['date'])  # old data
    # df_flow = pandas.read_csv("./data/sra_flow_aag2022.csv")                   # new data

    df_flow = pandas.read_csv("data/sd_flow_sra_v2_oct_data.csv", parse_dates=['date'])
    # most updated data 10-01-2022

    print("\nmin_date_flow:", min(df_flow["date"]), "max_date_flow:", max(df_flow["date"]))
    print("\nList of sra names in mobility data:", df_flow['sra_name'].unique())
    print("\nList the column names in mobility data:", df_flow.columns.values.tolist())

    for col in df_flow.columns.values.tolist():
        utils_pham.check_missing_rows(df_flow, col, "df_flow")

    return df_flow


if __name__ == "__main__":
    main()
    sys.exit()
