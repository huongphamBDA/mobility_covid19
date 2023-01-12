import os
import numpy
import pandas
import utils_pham
import method1_phases as tpp

from scipy import stats
from datetime import timedelta, date
from tslearn.metrics import dtw


def main():
    code_run_start, start_time = utils_pham.start_run_code()

    run_14days()

    utils_pham.print_code_run_time(code_run_start, start_time)


def run_14days():
    df_data, sra_names = tpp.import_and_process_data()
    sra_names_tolist = sorted(sra_names.tolist())

    case_n_flow_cols, flow_cols, flow_cols_short, cols_to_viz, phase_dates, phase_names = \
        utils_pham.define_flow_and_case_cols_and_phases()
    _, _, _, _, _, _, _, _, _, _, _, _, _, outdir_14days = utils_pham.create_output_folders()

    # test_normal_distribution(df_data, case_n_flow_cols)  # not work but not needed
    df_data_minmax = utils_pham.normalize_data(df_data)
    tpp.cal_netflow_and_7days_rolling_average(df_data_minmax)

    duration = 14
    dd_duration = timedelta(days=duration)

    dd_start = date(2020, 4, 1)
    dd_end = date(2021, 3, 1)
    day_total = int((dd_end - dd_start).days)
    column_names = ["Date"] + sra_names_tolist

    # WORK WITH CASE
    df_slope_case, df_intercept_case = get_slope_intercept_for_case(column_names, day_total, dd_duration, dd_start,
                                                                    df_data_minmax, duration, outdir_14days,
                                                                    sra_names_tolist)
    # add suffix to all column names except "Date" column
    df_slope_case = df_slope_case.add_suffix("_slope_case").rename(columns={"Date_slope_case": "Date"})
    df_intercept_case = df_intercept_case.add_suffix("_intercept_case").rename(columns={"Date_intercept_case": "Date"})

    df_merged_case = pandas.merge(df_slope_case, df_intercept_case, how="inner", on="Date")

    # WORK WITH FLOWS
    df_merged_flows_list = get_dtw_slope_intercept_for_flows(column_names, day_total, dd_duration, dd_start,
                                                             df_data_minmax, duration, flow_cols, flow_cols_short,
                                                             outdir_14days, sra_names_tolist)

    # COMBINE CASE AND FLOWS
    for ind, flow in enumerate(flow_cols):
        filename = "results_table_14days_no_shift"
        df_combined_case_flows = pandas.merge(df_merged_case, df_merged_flows_list[ind], how="inner", on="Date")

        get_results_14days(day_total, df_combined_case_flows, duration, flow, flow_cols_short, ind, outdir_14days,
                           sra_names_tolist, filename)


def get_results_14days(day_total, df_combined_case_flows, duration, flow, flow_cols_short, ind, outdir_14days,
                       sra_names_tolist, filename):
    get_labels(day_total, df_combined_case_flows, duration, flow_cols_short, ind, sra_names_tolist)

    # Sort dataframe columns in alphabetical order
    df_results_14days = df_combined_case_flows.reindex(sorted(df_combined_case_flows.columns), axis=1)

    # Put column "Date" back to the first column position
    first_column = df_results_14days.pop('Date')
    df_results_14days.insert(0, 'Date', first_column)

    df_results_14days.to_csv(os.path.join(outdir_14days, f"{filename}_{flow}.csv"), encoding="utf-8")
    save_to_html(df_results_14days, flow, outdir_14days, filename)

    return df_results_14days


def get_labels(day_total, df_combined_case_flows, duration, flow_cols_short, ind, sra_names_tolist):
    # GET THE LABELS
    for sra in sra_names_tolist:
        df_combined_case_flows[f"{sra}_label"] = ""

        for n in range(day_total - duration):
            slope_case = df_combined_case_flows.loc[n, f"{sra}_slope_case"]
            slope_flow = df_combined_case_flows.loc[n, f"{sra}_slope_{flow_cols_short[ind]}"]

            if slope_case > 0 and slope_flow > 0:
                col_label = "both increase"
            elif slope_case < 0 and slope_flow < 0:
                col_label = "both decrease"
            elif slope_case == 0 and slope_flow == 0:
                col_label = "both stable"
            else:
                col_label = "opposite"

            df_combined_case_flows.loc[n, f"{sra}_label"] = col_label


def get_dtw_slope_intercept_for_flows(column_names, day_total, dd_duration, dd_start, df_data_minmax, duration,
                                      flow_cols, flow_cols_short, outdir_14days, sra_names_tolist):
    df_merged_flow_list = []

    for idx, flow in enumerate(flow_cols):  # x5
        folder = f"{outdir_14days}/{flow}"

        if not os.path.exists(folder):
            os.makedirs(folder)

        df_dtw_flow = pandas.DataFrame(columns=column_names)
        df_slope_flow = pandas.DataFrame(columns=column_names)
        df_intercept_flow = pandas.DataFrame(columns=column_names)

        for n in range(day_total):  # x360
            # skip the first 13 days
            if n < dd_duration.days:
                continue
            else:
                mask_end = dd_start + timedelta(n)
                mask_start = mask_end - dd_duration
                mask_end_str = mask_end.strftime("%Y-%m-%d")
                mask_start_str = mask_start.strftime('%Y-%m-%d')
                mask = (df_data_minmax['date'] >= mask_start_str) & (df_data_minmax['date'] < mask_end_str)
                df_mask = df_data_minmax.loc[mask]

                each_row_dtw = [mask_end_str]
                each_row_slope = [mask_end_str]
                each_row_intercept = [mask_end_str]

                for sra in sra_names_tolist:  # x41
                    df_mask_sra = df_mask.loc[df_mask["name"] == sra]
                    dtw_score = dtw(df_mask_sra["cases_weekly_avg"], df_mask_sra[flow])
                    each_row_dtw.append(round(dtw_score, 1))

                    # Get the slope and intercept of the flow time series
                    array_days = numpy.array(range(duration))
                    array_flow = df_mask_sra[flow].to_numpy()

                    flow_regr_results = stats.linregress(array_days, array_flow)
                    flow_slope = round(flow_regr_results.slope, 3)
                    flow_intercept = round(flow_regr_results.intercept, 5)

                    each_row_slope.append(flow_slope)
                    each_row_intercept.append(flow_intercept)

                df_dtw_flow.loc[n] = each_row_dtw  # each_row is a list
                df_slope_flow.loc[n] = each_row_slope
                df_intercept_flow.loc[n] = each_row_intercept

        df_dtw_flow.to_csv(os.path.join(folder, f"DTW_table_14days_{flow}.csv"), encoding="utf-8")
        df_slope_flow.to_csv(os.path.join(folder, f"Slope_table_14days_{flow}.csv"), encoding="utf-8")
        df_intercept_flow.to_csv(os.path.join(folder, f"Intercept_table_14days_{flow}.csv"), encoding="utf-8")

        save_to_html(df_dtw_flow, flow, folder, "DTW_table_14days")
        save_to_html(df_slope_flow, flow, folder, "Slope_table_14days")
        save_to_html(df_intercept_flow, flow, folder, "Intercept_table_14days")

        # Concat three dataframes
        shortname = flow_cols_short[idx]
        df_dtw_flow = df_dtw_flow.add_suffix(f'_dtw_{shortname}').rename(columns={f"Date_dtw_{shortname}": "Date"})
        df_slope_flow = df_slope_flow.add_suffix(f"_slope_{shortname}").\
            rename(columns={f"Date_slope_{shortname}": "Date"})
        df_intercept_flow = df_intercept_flow.add_suffix(f"_intercept_{shortname}").\
            rename(columns={f"Date_intercept_{shortname}": "Date"})

        df_merged_flow = pandas.merge(pandas.merge(df_dtw_flow, df_slope_flow, how="inner", on="Date"),
                                      df_intercept_flow, how="inner", on="Date")
        df_merged_flow_list.append(df_merged_flow)

    return df_merged_flow_list


def get_slope_intercept_for_case(column_names, day_total, dd_duration, dd_start, df_data_minmax, duration,
                                 outdir_14days, sra_names_tolist):
    folder_case = f"{outdir_14days}/case"

    if not os.path.exists(folder_case):
        os.makedirs(folder_case)

    df_slope_case = pandas.DataFrame(columns=column_names)
    df_intercept_case = pandas.DataFrame(columns=column_names)

    for n in range(day_total):  # 360 days
        if n < dd_duration.days:  # skip the first 13 days
            continue
        else:
            mask_end = dd_start + timedelta(n)
            mask_start = mask_end - dd_duration
            mask_end_str = mask_end.strftime("%Y-%m-%d")
            mask_start_str = mask_start.strftime('%Y-%m-%d')
            mask = (df_data_minmax['date'] >= mask_start_str) & (df_data_minmax['date'] < mask_end_str)
            df_mask = df_data_minmax.loc[mask]

            each_row_slope = [mask_end_str]
            each_row_intercept = [mask_end_str]

            for sra in sra_names_tolist:  # x41
                df_mask_sra = df_mask.loc[df_mask["name"] == sra]

                # Get the slope and intercept of the case
                array_days = numpy.array(range(duration))
                array_case = df_mask_sra["cases_weekly_avg"].to_numpy()

                case_regr_results = stats.linregress(array_days, array_case)
                case_slope = round(case_regr_results.slope, 3)
                case_intercept = round(case_regr_results.intercept, 5)

                each_row_slope.append(case_slope)
                each_row_intercept.append(case_intercept)

            df_slope_case.loc[n] = each_row_slope
            df_intercept_case.loc[n] = each_row_intercept

    df_slope_case.to_csv(os.path.join(folder_case, f"Slope_table_14days_case.csv"), encoding="utf-8")
    df_intercept_case.to_csv(os.path.join(folder_case, f"Intercept_table_14days_case.csv"), encoding="utf-8")

    save_to_html(df_slope_case, "", folder_case, "Slope_table_14days_case")
    save_to_html(df_intercept_case, "", folder_case, "Intercept_table_14days_case")

    return df_slope_case, df_intercept_case


def save_to_html(df, flow, folder, name):
    df_to_html = df.to_html(escape=False, index=False, justify="center")
    fp = os.path.join(folder, f"{name}_{flow}.html")
    text_file = open(fp, "w")
    text_file.write(df_to_html)
    text_file.close()


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
