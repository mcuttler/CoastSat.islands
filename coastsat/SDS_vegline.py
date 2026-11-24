"""
SDS_vegline.py — Vegetation polygon extraction for small islands.

Extracts a vegetation polygon from satellite imagery using NDVI-based
thresholding, running in parallel to the sand polygon extraction in
SDS_islands.py.  The workflow mirrors extract_sand_poly() so that sand
and vegetation outputs share the same date index and can be directly
compared.

Adapted for CoastSat.islands from COASTGUARD (Muir et al., 2022):
  https://github.com/fmemuir/COASTGUARD

Authors:
  M. Cuttler  (University of Western Australia)  — islands integration
  F. Muir     (University of Glasgow)            — COASTGUARD core logic
"""

import os
import warnings
import numpy as np
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
from scipy.ndimage import binary_fill_holes
from shapely.geometry import LinearRing, Polygon

from coastsat import SDS_preprocess, SDS_tools

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# NDVI computation
# ---------------------------------------------------------------------------

def compute_ndvi(im_ms, cloud_mask):
    """
    Compute NDVI = (NIR − Red) / (NIR + Red).

    Band ordering (all sensors after CoastSat preprocessing):
        [Blue(0), Green(1), Red(2), NIR(3), SWIR1(4)]

    Parameters
    ----------
    im_ms : ndarray  (rows, cols, bands)
        Multi-spectral image returned by SDS_preprocess.preprocess_single.
    cloud_mask : ndarray  bool  (rows, cols)
        True where the pixel is cloud-contaminated.

    Returns
    -------
    im_ndvi : ndarray  float  (rows, cols)
        NDVI image; NaN at cloud-contaminated pixels.
    """
    nir = im_ms[:, :, 3].astype(float)
    red = im_ms[:, :, 2].astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        denom = nir + red
        im_ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    im_ndvi[cloud_mask] = np.nan
    return im_ndvi


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def threshold_ndvi(im_ndvi, cloud_mask, min_veg_area=500,
                   ndvi_floor=0.1, ndvi_ceiling=0.25):
    """
    Find the optimal NDVI threshold to separate vegetation from non-vegetation
    using Otsu's method on the valid (non-cloud) pixels.

    The Otsu threshold is clamped between ndvi_floor and ndvi_ceiling:
      - floor   (default 0.10): prevents sand/water from counting as vegetation
                                when Otsu lands too low.
      - ceiling (default 0.25): prevents sparse vegetation from being missed
                                when the NDVI histogram is dominated by sand
                                and Otsu climbs too high.

    Neither bound affects the sand/water NN classifier — this function is
    called only for vegetation detection.

    Parameters
    ----------
    im_ndvi : ndarray  float
        NDVI image (NaN at clouds).
    cloud_mask : ndarray  bool
        Cloud mask.
    min_veg_area : int
        Minimum vegetated pixel count required; returns empty mask if fewer.
    ndvi_floor : float
        Minimum allowed Otsu threshold (default 0.1).
    ndvi_ceiling : float
        Maximum allowed Otsu threshold (default 0.25).

    Returns
    -------
    thresh : float
        NDVI threshold used.
    im_veg : ndarray  bool
        Binary vegetation mask (True = vegetated pixel).
    """
    valid = ~cloud_mask & ~np.isnan(im_ndvi)
    ndvi_vals = im_ndvi[valid]

    if ndvi_vals.size == 0:
        return np.nan, np.zeros(im_ndvi.shape, dtype=bool)

    try:
        thresh = float(filters.threshold_otsu(ndvi_vals))
        thresh = max(thresh, ndvi_floor)    # floor:   don't go below this
        thresh = min(thresh, ndvi_ceiling)  # ceiling: don't go above this
    except Exception:
        thresh = ndvi_floor                 # safe fallback

    im_veg = (im_ndvi > thresh) & valid

    if np.sum(im_veg) < min_veg_area:
        return thresh, np.zeros(im_ndvi.shape, dtype=bool)

    return thresh, im_veg


# ---------------------------------------------------------------------------
# Core polygon detection
# ---------------------------------------------------------------------------

def find_veg_polygon(im_ms, cloud_mask, settings, sand_mask=None):
    """
    Extract the vegetation polygon from a multi-spectral image.

    Steps:
      1. Compute NDVI.
      2. Threshold (auto Otsu or fixed value from settings).
      3. Optionally constrain to sand_mask (prevents offshore NDVI bleed).
      4. Morphological cleaning (remove small objects, fill holes).
      5. Find the largest contour → pixel-space contour (row, col).

    Parameters
    ----------
    im_ms : ndarray  (rows, cols, bands)
    cloud_mask : ndarray  bool
    settings : dict
        Relevant keys:
          'min_veg_area'       : int   minimum vegetated area in pixels (default 500)
          'veg_thresh'         : float or 'auto'   NDVI threshold (default 'auto')
          'ndvi_floor'         : float  Otsu lower clamp (default 0.10)
          'ndvi_thresh_ceiling': float  Otsu upper clamp (default 0.25)
    sand_mask : ndarray bool or None
        When provided, NDVI hits outside the sand polygon are discarded.
        Prevents vegetation being detected in surf / open water.

    Returns
    -------
    veg_contour : ndarray (N, 2) or None
        Outer boundary in pixel coordinates (row, col).
        None if no vegetation was detected.
    ndvi_thresh : float
        NDVI threshold that was applied.
    im_veg_binary : ndarray bool
        Cleaned binary vegetation mask (same shape as cloud_mask).
    """
    min_veg_area = int(settings.get('min_veg_area', 500))
    veg_thresh_setting = settings.get('veg_thresh', 'auto')

    im_ndvi = compute_ndvi(im_ms, cloud_mask)

    if veg_thresh_setting == 'auto':
        ndvi_thresh, im_veg = threshold_ndvi(
            im_ndvi, cloud_mask, min_veg_area,
            ndvi_floor=settings.get('ndvi_floor', 0.1),
            ndvi_ceiling=settings.get('ndvi_thresh_ceiling', 0.25),
        )
    else:
        ndvi_thresh = float(veg_thresh_setting)
        valid = ~cloud_mask & ~np.isnan(im_ndvi)
        im_veg = (im_ndvi > ndvi_thresh) & valid

    # --- constrain to sand polygon (prevents offshore NDVI bleed) ---
    if sand_mask is not None:
        im_veg = im_veg & sand_mask

    _empty = np.zeros(cloud_mask.shape, dtype=bool)

    if not np.any(im_veg):
        return None, ndvi_thresh, _empty

    # --- morphological cleaning ---
    im_veg = morphology.remove_small_objects(im_veg, min_size=min_veg_area)
    im_veg = binary_fill_holes(im_veg)
    im_veg = morphology.remove_small_holes(im_veg, area_threshold=min_veg_area)

    if not np.any(im_veg):
        return None, ndvi_thresh, _empty

    # --- outer contour ---
    contours = measure.find_contours(im_veg.astype(float), 0.5)
    if not contours:
        return None, ndvi_thresh, im_veg

    main_contour = contours[np.argmax([len(c) for c in contours])]

    return main_contour, ndvi_thresh, im_veg


# ---------------------------------------------------------------------------
# Unclassified-pixel vegetation detection
# ---------------------------------------------------------------------------

def find_veg_unclassified(im_labels, im_binary_sand_closed, cloud_mask, settings):
    """
    Detect vegetation as NN-unclassified pixels inside the sand polygon.

    The CoastSat NN classifier labels each pixel as sand, whitewater, or water
    (im_labels planes 0, 1, 2 respectively).  Pixels that are inside the sand
    polygon but not assigned to any of those three classes are spectrally
    distinct — almost always vegetation on a sandy island.

    This is a pragmatic interim approach.  The long-term solution is to add
    vegetation as an explicit class to the NN classifier.

    Parameters
    ----------
    im_labels : ndarray  bool  (rows, cols, 3)
        NN classification planes: [:,:,0]=sand, [:,:,1]=whitewater, [:,:,2]=water.
    im_binary_sand_closed : ndarray  bool
        Filled sand polygon mask (True = inside island footprint).
    cloud_mask : ndarray  bool
    settings : dict
        Relevant key: 'min_veg_area' (int, default 500 pixels).

    Returns
    -------
    veg_contour : ndarray (N, 2) or None
        Outer boundary in pixel coordinates (row, col).
    im_veg_binary : ndarray  bool
        Cleaned binary vegetation mask.
    """
    min_veg_area = int(settings.get('min_veg_area', 500))

    # pixels the NN assigned to any known class
    im_classified = im_labels[:, :, 0] | im_labels[:, :, 1] | im_labels[:, :, 2]

    # unclassified pixels inside the island, excluding cloud
    im_veg = im_binary_sand_closed & ~im_classified & ~cloud_mask

    _empty = np.zeros(cloud_mask.shape, dtype=bool)

    if not np.any(im_veg):
        return None, _empty

    # --- morphological cleaning ---
    im_veg = morphology.remove_small_objects(im_veg, min_size=min_veg_area)
    im_veg = binary_fill_holes(im_veg)
    im_veg = morphology.remove_small_holes(im_veg, area_threshold=min_veg_area)

    if not np.any(im_veg):
        return None, _empty

    # --- outer contour ---
    contours = measure.find_contours(im_veg.astype(float), 0.5)
    if not contours:
        return None, im_veg

    main_contour = contours[np.argmax([len(c) for c in contours])]

    return main_contour, im_veg


# ---------------------------------------------------------------------------
# Unclassified-pixel vegetation detection (default)
# ---------------------------------------------------------------------------

def find_veg_unclassified(im_labels, im_binary_sand_closed, cloud_mask, settings):
    """
    Detect vegetation as NN-unclassified pixels inside the sand polygon.

    The CoastSat NN classifier labels each pixel as sand, whitewater, or water
    (im_labels planes 0, 1, 2 respectively).  Pixels that are inside the sand
    polygon but not assigned to any of those three classes are spectrally
    distinct — almost always vegetation on a sandy island.

    This is a pragmatic interim approach.  The long-term solution is to add
    vegetation as an explicit class to the NN classifier.

    Parameters
    ----------
    im_labels : ndarray  bool  (rows, cols, 3)
        NN classification planes: [:,:,0]=sand, [:,:,1]=whitewater, [:,:,2]=water.
    im_binary_sand_closed : ndarray  bool
        Filled sand polygon mask (True = inside island footprint).
    cloud_mask : ndarray  bool
    settings : dict
        Relevant key: 'min_veg_area' (int, default 500 pixels).

    Returns
    -------
    veg_contour : ndarray (N, 2) or None
        Outer boundary in pixel coordinates (row, col).
    im_veg_binary : ndarray  bool
        Cleaned binary vegetation mask.
    """
    min_veg_area = int(settings.get('min_veg_area', 500))

    # pixels the NN assigned to any known class
    im_classified = im_labels[:, :, 0] | im_labels[:, :, 1] | im_labels[:, :, 2]

    # unclassified pixels inside the island, excluding cloud
    im_veg = im_binary_sand_closed & ~im_classified & ~cloud_mask

    _empty = np.zeros(cloud_mask.shape, dtype=bool)

    if not np.any(im_veg):
        return None, _empty

    # --- morphological cleaning ---
    im_veg = morphology.remove_small_objects(im_veg, min_size=min_veg_area)
    im_veg = binary_fill_holes(im_veg)
    im_veg = morphology.remove_small_holes(im_veg, area_threshold=min_veg_area)

    if not np.any(im_veg):
        return None, _empty

    # --- outer contour ---
    contours = measure.find_contours(im_veg.astype(float), 0.5)
    if not contours:
        return None, im_veg

    main_contour = contours[np.argmax([len(c) for c in contours])]

    return main_contour, im_veg


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_veg_metrics(veg_polygon_world, im_veg_binary):
    """
    Compute shape metrics for a vegetation polygon.

    Area, perimeter and centroid are taken directly from veg_polygon_world
    (which is already in world/projected coordinates, metres).  Eccentricity
    and orientation come from skimage regionprops on the pixel binary mask,
    matching the convention used for the sand polygon metrics.

    Parameters
    ----------
    veg_polygon_world : shapely.Polygon in world coordinates (output_epsg, metres)
        or None if no vegetation was detected.
    im_veg_binary : ndarray  bool  (rows, cols)

    Returns
    -------
    metrics : dict
        'veg_area'          : float  m²
        'veg_perimeter'     : float  m
        'veg_centroid'      : [float, float]  (easting, northing) in output_epsg
        'veg_eccentricity'  : float  (0 = circle, 1 = line)
        'veg_orientation'   : float  radians from positive x-axis (same convention as sand)
    """
    nan2 = [np.nan, np.nan]
    metrics = dict(veg_area=np.nan, veg_perimeter=np.nan,
                   veg_centroid=nan2, veg_eccentricity=np.nan,
                   veg_orientation=np.nan)

    if veg_polygon_world is None or veg_polygon_world.is_empty:
        return metrics

    # --- area, perimeter, centroid directly from world polygon ---
    try:
        metrics['veg_area'] = float(veg_polygon_world.area)
        metrics['veg_perimeter'] = float(veg_polygon_world.exterior.length)
        cx, cy = veg_polygon_world.centroid.coords[0]
        metrics['veg_centroid'] = [float(cx), float(cy)]
    except Exception:
        pass

    # --- eccentricity and orientation from pixel binary (matches sand convention) ---
    try:
        labeled = measure.label(im_veg_binary.astype(int))
        props = measure.regionprops(labeled)
        if props:
            largest = max(props, key=lambda p: p.area)
            metrics['veg_eccentricity'] = float(largest.eccentricity)
            metrics['veg_orientation'] = float(largest.orientation)
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_veglines_islands(metadata, settings):
    """
    Extract vegetation polygons from all satellite images for an island site.

    Loops through every image in ``metadata`` (same structure produced by
    ``SDS_download.retrieve_images``), preprocesses each one, computes NDVI,
    and extracts the outer vegetation boundary as a polygon together with
    planform metrics.  Designed to be called alongside
    ``SDS_islands.extract_sand_poly`` with the same ``metadata`` and
    ``settings`` dicts.

    Parameters
    ----------
    metadata : dict
        Output of ``SDS_download.retrieve_images`` / ``SDS_download.get_metadata``.
    settings : dict
        Standard CoastSat.islands settings dict.  Additional keys:

        'min_veg_area' : int
            Minimum vegetated area in pixels.  Default 500.
        'veg_thresh' : float or 'auto'
            NDVI threshold.  'auto' uses Otsu's method.  Default 'auto'.
        'cloud_thresh' : float
            Maximum cloud-cover fraction (0–1).  Default 0.5.

    Returns
    -------
    output : dict
        One entry per satellite name, each a dict of parallel lists:

        'dates'              — datetime objects (UTC)
        'filename'           — image filenames
        'cloud_cover'        — cloud cover fraction
        'veg_polygon_coords' — Nx2 array of (easting, northing) exterior coords
                               in the output CRS, or None if not detected
        'veg_area'           — polygon area (m²)
        'veg_perimeter'      — perimeter (m)
        'veg_centroid'       — [easting, northing] of centroid
        'veg_eccentricity'   — eccentricity (0 = circle, 1 = line)
        'veg_orientation'    — orientation of major axis (radians)
        'ndvi_threshold'     — NDVI threshold applied
        'geoaccuracy'        — georeferencing accuracy (m)

    Example
    -------
    >>> from coastsat import SDS_vegline
    >>> veg_output = SDS_vegline.extract_veglines_islands(metadata, settings)
    >>> # Combine with sand output from SDS_islands.extract_sand_poly()
    """

    settings.setdefault('min_veg_area', 500)
    settings.setdefault('veg_thresh', 'auto')
    settings.setdefault('cloud_thresh', 0.5)

    output = {}

    for satname in metadata.keys():

        output[satname] = dict(
            dates=[],
            filename=[],
            cloud_cover=[],
            veg_polygon_coords=[],
            veg_area=[],
            veg_perimeter=[],
            veg_centroid=[],
            veg_eccentricity=[],
            veg_orientation=[],
            ndvi_threshold=[],
            geoaccuracy=[],
        )

        filenames = metadata[satname]['filenames']
        n_images = len(filenames)

        if n_images == 0:
            print(f'  {satname}: no images available')
            continue

        print(f'\nVegetation extraction — {satname}: {n_images} images')

        filepath = SDS_tools.get_filepath(settings['inputs'], satname)

        for i in range(n_images):
            print('\r  %s:  %d%%' % (satname, int(((i + 1) / n_images) * 100)),
                  end='')

            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            image_epsg = metadata[satname]['epsg'][i]

            # --- preprocessing (same call as extract_sand_poly after the fix) ---
            try:
                im_ms, georef, cloud_mask, im_extra, imQA, im_nodata = \
                    SDS_preprocess.preprocess_single(
                        fn, satname,
                        settings['cloud_mask_issue'],
                        settings['pan_off'])
            except Exception as exc:
                print(f'\n    [{i + 1}/{n_images}] preprocessing error: {exc}')
                continue

            # --- cloud filter ---
            cloud_cover = float(np.sum(cloud_mask) / cloud_mask.size)
            if cloud_cover > settings['cloud_thresh']:
                continue

            # --- extract vegetation polygon ---
            veg_polygon, ndvi_thresh = find_veg_polygon(im_ms, cloud_mask, settings)

            # --- metrics ---
            if veg_polygon is not None:
                # Reconstruct binary mask for regionprops (consistent with polygon)
                im_ndvi = compute_ndvi(im_ms, cloud_mask)
                t = ndvi_thresh if np.isfinite(ndvi_thresh) else 0.3
                im_veg_bin = binary_fill_holes(
                    (im_ndvi > t) & ~cloud_mask & ~np.isnan(im_ndvi)
                )
                metrics = compute_veg_metrics(veg_polygon, im_veg_bin, georef)

                # Exterior coords in world CRS for storage
                ext = np.array(veg_polygon.exterior.coords)
                x_w = georef[0] + ext[:, 1] * georef[1]
                y_w = georef[3] + ext[:, 0] * georef[5]
                veg_coords = np.column_stack([x_w, y_w])
                # reproject from native image CRS to the target output CRS
                veg_coords = SDS_tools.convert_epsg(veg_coords, image_epsg,
                                                    settings['output_epsg'])
                # also reproject the centroid
                if not np.isnan(metrics['veg_centroid'][0]):
                    cen = SDS_tools.convert_epsg(
                        np.array([metrics['veg_centroid']]),
                        image_epsg, settings['output_epsg'])
                    metrics['veg_centroid'] = list(cen[0])
            else:
                metrics = dict(veg_area=np.nan, veg_perimeter=np.nan,
                               veg_centroid=[np.nan, np.nan],
                               veg_eccentricity=np.nan,
                               veg_orientation=np.nan)
                veg_coords = None

            # --- store ---
            output[satname]['dates'].append(metadata[satname]['dates'][i])
            output[satname]['filename'].append(filenames[i])
            output[satname]['cloud_cover'].append(cloud_cover)
            output[satname]['veg_polygon_coords'].append(veg_coords)
            output[satname]['veg_area'].append(metrics['veg_area'])
            output[satname]['veg_perimeter'].append(metrics['veg_perimeter'])
            output[satname]['veg_centroid'].append(metrics['veg_centroid'])
            output[satname]['veg_eccentricity'].append(metrics['veg_eccentricity'])
            output[satname]['veg_orientation'].append(metrics['veg_orientation'])
            output[satname]['ndvi_threshold'].append(ndvi_thresh)
            output[satname]['geoaccuracy'].append(
                metadata[satname]['acc_georef'][i])

        print()   # newline after progress bar

    return output
