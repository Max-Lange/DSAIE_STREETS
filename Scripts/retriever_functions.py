from ast import literal_eval
import geopandas as gpd
import pandas as pd
import numpy as np
from IPython.display import Image, display

def get_coupled_panoid_images(section_index: int, panoid_gdf: gpd.GeoDataFrame,
                              coupling: pd.DataFrame, data_folder_path: str,
                              display_front: bool=False):
    if coupling.loc[section_index, 'coupled_panoids'] is np.NaN:
        return ValueError(f"Section {section_index} has no coupled panoids")

    coupled_panoids = literal_eval(coupling.loc[section_index, 'coupled_panoids'])

    images_dict = {}
    for panoid_index in coupled_panoids:
        panoid_x_dict = {}
        for side in ['side_a', 'front', 'side_b', 'back']:
            panoid_x_dict[side] = f"{data_folder_path}\\Delft_NL\\imagedb\\{panoid_gdf.loc[panoid_index, 'im_'+side]}"
        images_dict[panoid_index] = panoid_x_dict

    if display_front:
        for panoid_index in coupled_panoids:
            panoid = panoid_gdf.loc[panoid_index]
            display(Image(data_folder_path + r"\\Delft_NL\\imagedb\\" + panoid["im_front"]))

    return images_dict