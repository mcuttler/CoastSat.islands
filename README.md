# CoastSat.islands

Satellite-derived shorelines and 2D planform measurements for islands, extension of the [CoastSat toolbox](https://github.com/kvos/CoastSat).
![](./doc/Eva_sand_polygons.gif)
This toolkit enables users to measure shoreline position and two-dimensional planform characteristics (area and orientation) for small sandy islands. It has the following functionalities:
- Download satellites images from Google Earth Engine (as in CoastSat)
- Map the island contours as polygons
- Tidally correct the island contours
- Compute island metrics such as area and orientation.

Detailed methodology and application are described in: *Cuttler MVW, Vos K, Branson P, Hansen JE, O'Leary M, Browne NK, Lowe RJ (2020) Reef island response to climate-driven variations in water level and wave climate (under review).*

![](./doc/Eva_area.gif)

To run this toolkit you will need to have the `coastsat` environment installed (instructions in the main [CoastSat toolbox](https://github.com/kvos/CoastSat)).

The [Jupyter Notebook]((https://github.com/mcuttler/CoastSat.islands/blob/master/example_island_Eva_Island.ipynb) in the repository shows an example of satellite-derived shorelines and island area estimation at Eva Island, Western Australia. There is also a [Python script](https://github.com/mcuttler/CoastSat.islands/blob/master/example_islands.py) for users who prefer to use Spyder/PyCharm.

For the tidal correction, you will need time-series of water/tide levels at your site. You can provide those in a .txt file as shown in the /example folder. Otherwise, you can use a global tide model to get the modeled tide levels. To use [FES2014](https://www.aviso.altimetry.fr/es/data/products/auxiliary-products/global-tide-fes/description-fes2014.html) global tide model to get the tide levels at the time of image acquisition, refer to the [CoastSat.slope](https://github.com/kvos/CoastSat.slope) repository and follow the [instructions](https://github.com/kvos/CoastSat.slope/blob/master/doc/FES2014_installation.md) provided to setup the tide model.

**If you like the repo put a star on it!**

Having a problem? Post an issue in the [Issues page](https://github.com/mcuttler/CoastSat.islands/issues) (please do not email).
