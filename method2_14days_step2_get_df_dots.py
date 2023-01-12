import numpy
import pandas
import plotly.express as px
import method1_phases as tpp
import plotly.graph_objects as go
import dataframe_image as dfi
# import plotly.figure_factory as ff

from functools import reduce


def main():
    _, sra_num, sra_name, df_sra_info = tpp.import_covid_data()

    dict1 = create_dict()
    num = len(dict1["filenames"])
    corr_count_list = [sra_name.tolist()]
    column_names = ["SRA names"] + dict1['keywords']

    # go through each file in the five results files
    for i in range(num):
        df = pandas.read_csv(f"output_method2_14days/{dict1['filenames'][i]}.csv")
        # eg., df = pandas.read_csv("./output_method2_14days/results_table_14days_no_shift_inflow_weekly_avg.csv")

        dtw_max_list = []
        dtw_min_list = []
        df_sra_list = []

        flow_shortened = dict1['short'][i]
        flow = dict1['keywords'][i]

        # go though each SRA name
        for sra in sra_name:
            sra_cols = [col for col in df.columns if (sra in col and col.startswith(sra))]  # col is column name

            sra_cols.insert(0, "Date")  # insert Date into the first position of the list
            df_sra = df[sra_cols]       # df of each sra with Date column

            # approach 2: replace df_sra's dtw values to "" for those having label different from increase-increase
            # 10 is a dummy value and will be removed after the using the condition in row 52
            # remove only the following row and uncomment the min, mix of dtw section to come back to approach 1
            df_sra.loc[df_sra[f"{sra}_label"] != "both increase", f"{sra}_dtw_{flow_shortened}"] = 10
            df_sra.loc[df_sra[f"{sra}_dtw_{flow_shortened}"] > 1.0, f"{sra}_dtw_{flow_shortened}"] = 1.0  # change 1/2

            # max, min of dtw?
            # dtw_max = df_sra.loc[df_sra[f"{sra}_dtw_{flow_shortened}"].idxmax()][f"{sra}_dtw_{flow_shortened}"]
            # dtw_max_list.append(dtw_max)
            #
            # dtw_min = df_sra.loc[df_sra[f"{sra}_dtw_{dict1['short'][i]}"].idxmin()][f"{sra}_dtw_{dict1['short'][i]}"]
            # dtw_min_list.append(dtw_min)

            # approach 1: get df_sra with two conditions
            # df_sra = df_sra[(df_sra[f"{sra}_label"] == "both increase") &
            #                 (df_sra[f"{sra}_dtw_{dict1['short'][i]}"] < 1.0)]

            df_sra_list.append(df_sra)

        # --------
        # max dtw of all 41 sras for each type of flow (= max in results_table_14days_no_shift_inflow_weekly_avg)
        # dtw_max_41sra = max(dtw_max_list)
        # # min dtw of all 41 sras (similar as above)
        # dtw_min_41sra = min(dtw_min_list)
        # print(f"for {flow}, max dtw of all 41 SRAs = {dtw_max_41sra}, min dtw of all 41 SRAs = {dtw_min_41sra}")
        # e.g., for inflow, max dtw of all 41 SRAs = 2.8, min dtw of all 41 SRAs = 0.0

        # --------
        # merge 41 df_sra to one
        df_merged = reduce(lambda left, right: pandas.merge(left, right, on=['Date'],
                                                            how='outer'), df_sra_list).fillna("")
        df_merged = df_merged.sort_values(by=['Date'])

        # keep only dtw columns for 41 SRAs:
        new_cols = [col for col in df_merged.columns if "dtw" in col]
        new_cols.insert(0, "Date")
        df_dots = df_merged[new_cols]
        # read top of this script about df_dots

        # --------
        # Count days of correlation for each SRA
        df_dots_nan = df_dots.replace(r'^\s*$', numpy.nan, regex=True)
        df_count_correlation = df_dots_nan.count().tolist()
        corr_count_list.append(df_count_correlation[1:])

        # --------
        # Save to csv and html
        # df_dots.to_csv(f"./output_method2_14days/df_dots_{flow}.csv")  # approach 1
        df_dots.to_csv(f"./outdir_14days_noshift/df_dots_320days_{flow}.csv")  # approach 2

        df_to_html = df_dots.to_html(escape=False, index=False, justify="center")
        # fp = f"./output_method2_14days/df_dots_{flow}.html"  # approach 1
        fp = f"./outdir_14days_noshift/df_dots_320days_{flow}.html"  # approach 2
        # df_dots are df of all 41 SRAs that case & flow moved in same direction (increase) and have dtw less than 1.0
        text_file = open(fp, "w")
        text_file.write(df_to_html)
        text_file.close()

        # --------
        # Create a new dataframe from transposed df_dots
        df_dots_transposed = df_dots.set_index("Date").T
        # style this table so that column names become vertical to save space
        # df_dots_transposed.to_csv(f"./output_method2_14days/df_dots_{flow}_transposed.csv")  # approach 1
        df_dots_transposed.to_csv(f"./outdir_14days_noshift/df_dots_320days_{flow}_transposed.csv")  # approach 2

        # --------
        # empty strings/missing values converted to 1.0
        df_dots_transposed = df_dots_transposed.replace("", 1.0)  # change 2/2

        plot_heatmap_for_df_dots(df_dots_transposed, flow, flow_shortened)

    df_corr_count = pandas.DataFrame(corr_count_list)
    df_corr_count = df_corr_count.transpose()
    df_corr_count.set_axis(column_names, axis=1, inplace=True)
    df_corr_count_sra_nums = df_corr_count.merge(df_sra_info, how='left', left_on=["SRA names"], right_on=['sra_name'])
    df_corr_count_sra_nums = df_corr_count_sra_nums.drop(['sra_name'], axis=1)
    df_corr_count_sra_nums = df_corr_count_sra_nums[["sra_num", "SRA names", "inflow", "netflow", "outflow",
                                                     "inwithinflow", "withinflow"]]
    df_corr_count_sra_nums.rename(columns={"sra_num": "SRA Numbers", "SRA names": "SRA Names"}, inplace=True)

    df_corr_count.to_csv(f"./outdir_14days_noshift/df_count_correlation.csv")
    df_corr_count_sra_nums.to_csv(f"./outdir_14days_noshift/df_count_correlation_with_sra_nums.csv")

    df_to_html2 = df_corr_count.to_html(escape=False, index=False, justify="center")
    fp = f"./outdir_14days_noshift/df_count_correlation.html"
    # df_dots are df of all 41 SRAs that case & flow moved in same direction (increase) and have dtw less than 1.0
    text_file = open(fp, "w")
    text_file.write(df_to_html2)
    text_file.close()


def plot_heatmap_for_df_dots(df_dots_transposed, flow, flow_shortened):
    # --------
    # Create heatmap for each of the df_dots dataframes
    # https://plotly.com/python/annotated-heatmap/
    z = df_dots_transposed.to_numpy()
    # z_text = numpy.around(z, decimals=2)

    # x is a list of days to show in the heatmap
    x = list(df_dots_transposed.columns.values)
    # x1 = x[:len(x) // 2]
    # x2 = x[len(x) // 2:]

    # y is a list of SRA names to show in heatmap (need to remove _dtw_in, _x, _y from y_temp)
    y_temp = list(df_dots_transposed.index.values)
    y = list({element.replace(f"_dtw_{flow_shortened}", "") for element in y_temp})

    # --- plotly express no x, y  - looks better than the other below
    # fig = px.imshow(z, text_auto=True)
    # fig.show()

    # --- plotly express with x, y
    # fig = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', text_auto=".2f", aspect="auto")  # Greys_r,
    # fig.update_xaxes(tickangle=90, title_font_family="Arial", side="bottom")
    # fig.write_image(f"./output_method2_14days/heatmap_14days_df_dots_{flow}.png")
    # fig.show()

    # --- plotly go
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Oranges_r'))  # colorscale='Viridis' 'Oranges_r'

    fig.update_layout(
        title=f"HEATMAP DTW VALUES, 14 DAY APPROACH -- {flow}",
        xaxis_title="DATE",
        yaxis_title="SAN DIEGO SRA's NAMES",
        legend_title="DTW Values",
        font=dict(
            family="Courier New, monospace",
            size=10
        )  # color="RebeccaPurple"
    )
    # fig.write_image(f"./output_method2_14days/heatmap_14days_df_dots_{flow}.png")  # approach 1
    fig.write_image(f"./outdir_14days_noshift/heatmap_14days_df_dots_320days_{flow}.png")  # approach 2
    # fig.show()

    # --- figure factory
    # fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Greys_r', hoverinfo='z')  # annotation_text=z_text,
    # fig.update_xaxes(tickangle=90, title_font_family="Arial", side="bottom")
    # fig.layout.autosize = True
    # fig.show()

    # file_path = f"./output_method2_14days/heatmap_14days_df_dots_{flow}.html"  # approach 1
    file_path = f"output_method2_14days/heatmap_14days_df_dots_320days_{flow}.html"  # approach 2
    # (having all same number of colunms)
    fig.write_html(file=file_path, include_plotlyjs="cdn")
    # -------- end


def format_vertical_headers(df):
    """Display a dataframe with vertical column headers"""

    styles = [dict(selector="th", props=[('width', '40px')]),
              dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(180deg)'),
                          ('height', '290px'),
                          ('vertical-align', 'top')])]

    return df.fillna('').style.set_table_styles(styles)


def create_dict():
    dictionary = {"filenames": ["results_table_14days_no_shift_inflow_weekly_avg",
                                "results_table_14days_no_shift_netflow_weekly_avg",
                                "results_table_14days_no_shift_outflow_weekly_avg",
                                "results_table_14days_no_shift_total_in_within_weekly_avg",
                                "results_table_14days_no_shift_withinflow_weekly_avg"],
                  "short": ["in", "net", "out", "inwithin", "within"],
                  "keywords": ["inflow", "netflow", "outflow", "inwithinflow", "withinflow"]}

    return dictionary


if __name__ == "__main__":
    main()
