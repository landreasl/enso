# %%
import os
import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
# %%

dt = xr.open_dataset("era5_sst_2000-2019_mon_anomalies_coars.nc")
print(dt)

''
fig = plt.figure(figsize=(15,15))
m1 = fig.add_subplot(projection=ccrs.PlateCarree())
m1.add_feature(cfeature.LAND)
m1.add_feature(cfeature.OCEAN)

m1.stock_img()
# %%
