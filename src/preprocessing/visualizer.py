import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from config import Config


class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.vn_adm1 = None
        self.dbscl = None

    def load_shapefile(self):
        if self.vn_adm1 is None:
            self.vn_adm1 = gpd.read_file(self.config.vn_adm1_path)
            self.dbscl = self.vn_adm1[self.vn_adm1["NAME_1"].isin(self.config.mekong_provinces)]

    def plot_map(self, x: np.ndarray, title: str = "Rainfall Prediction Map", cmap: str = 'turbo'):
        self.load_shapefile()

        extent = [self.config.min_lon, self.config.max_lon, self.config.min_lat, self.config.max_lat]

        fig, ax = plt.subplots(figsize=(10, 10))

        self.vn_adm1.boundary.plot(ax=ax, color='lightgrey', linewidth=0.4)
        self.dbscl.boundary.plot(ax=ax, color='black', linewidth=1)

        im = ax.imshow(x, extent=extent, origin='lower', cmap=cmap, alpha=0.8)

        plt.colorbar(im, ax=ax, label="Rainfall prediction (mm/h)")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(self.config.min_lon, self.config.max_lon)
        ax.set_ylim(self.config.min_lat, self.config.max_lat)
        plt.tight_layout()
        plt.show()