# fixed for all flows (from only inflow) at 12:35 am 11/22/2022
import os
import shutil
import time
import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import method2_14days_step2_get_df_dots as dd

from PIL import Image


def main():
    data_geo_dir = "./data/geo"
    fp_sra = os.path.join(data_geo_dir, 'acs_sra_2013_wgs84_v2.shp')
    gdf_sra = gpd.GeoDataFrame.from_file(fp_sra)

    dict = dd.create_dict()
    num = len(dict["filenames"])

    directory = "./output_method2_14days"

    for i in range(num):
        short_i = dict['short'][i]
        flow_i = dict['keywords'][i]

        fn = f"df_dots_320days_{flow_i}.csv"
        fp = directory + "/" + fn

        print(f"\ncreating gif for {flow_i} ...")

        make_gif(directory, fp, flow_i, short_i, gdf_sra)


def make_gif(directory, fp, flow, short, gdf_sra):
    """https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python"""

    # create directories: out_dir = "./output_method2_14days/gif_output"
    out_dir = directory + "/_gif_output"
    os.makedirs(out_dir, exist_ok=True)
    file_format = '.png'

    # create directory for flows: eg "./output_method2_14days/gif_output/flow"
    out_dir_flow = out_dir + "/" + flow
    os.makedirs(out_dir_flow, exist_ok=True)

    # create a tmp directory, eg "./output_method2_14days/gif_output/flow/_tmp"
    out_dir_tmp = out_dir_flow + "/_tmp"
    os.makedirs(out_dir_tmp, exist_ok=True)

    # choropleth map parameters
    font_size_label = 7
    title_coords = (-116.6, 33.5)  # (-116.4, 33.5)
    font_size_title = 12

    # gif parameters
    wait_time = 0.00
    gif_duration = 600

    # df_data
    df_data = pd.read_csv(fp)
    df_data = df_data.rename(columns=lambda x: x[:-(len(f"_dtw_{short}"))] if x.endswith(f"_dtw_{short}") else x)
    df_data = df_data.fillna("")

    # choropleth map for each day
    for r_index, row in df_data.iterrows():
        date = row[1]

        if date == '2020-09-07' or date == '2020-12-18':
            continue

        sras = row[2:].reset_index().rename({'index': 'sra', r_index: 'dtw'}, axis=1)  # series horizontal to vertical

        print('{}'.format(date))

        outfile_path = os.path.join(out_dir_tmp, f'{flow.lower()}_{date}{file_format}')

        # merge each day DTW with sra GeoDataFrame
        gdf_sra_merge = gdf_sra.merge(sras, left_on='NAME', right_on='sra')

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_aspect('equal')
        ax.set_xlabel("\nSan Diego SRA's longitude")
        ax.set_ylabel("San Diego SRA's latitude\n")

        gdf_sra_merge.plot(ax=ax, column='dtw', cmap='Blues_r', legend=True, edgecolor="Grey")
        gdf_sra_merge['coords'] = gdf_sra_merge['geometry'].apply(lambda x: x.representative_point().coords[:])
        gdf_sra_merge['coords'] = [coords[0] for coords in gdf_sra_merge['coords']]
        gdf_sra_merge['NAME'] = gdf_sra_merge['NAME'].str.title()

        for idx, r in gdf_sra_merge.iterrows():
            plt.annotate(text=r['NAME'], xy=r['coords'], color='darkslategray', fontsize=font_size_label,
                         horizontalalignment='left', verticalalignment='top')  # , rotation=35

        plt.annotate(text=f'{flow}: {date}', color='black', fontsize=font_size_title,
                     xy=title_coords, horizontalalignment='center')
        # plt.legend(title='DTW values')
        plt.title(f"DTW values of SRAs in days that Covid-19 case and human mobility both increase, {flow}\n")

        # plt.show()
        plt.savefig(outfile_path)
        plt.clf()
        plt.close()
        time.sleep(wait_time)

    # create gif
    out_file_path_gif = os.path.join(out_dir_flow, "{}.gif".format(flow.lower()))

    frames = [Image.open(image) for image in sorted(glob.glob(f"{out_dir_tmp}/*{file_format}"))]
    frame_one = frames[0]
    frame_one.save(out_file_path_gif, format="GIF", append_images=frames, save_all=True, duration=gif_duration)

    # remove the whole directory _tmp after making GIF
    FLAG_REMOVE_TMP_IMAGES = False
    if FLAG_REMOVE_TMP_IMAGES:
        shutil.rmtree(out_dir_tmp)


if __name__ == "__main__":
    main()
