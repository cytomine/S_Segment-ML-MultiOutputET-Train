import os
from joblib import Parallel, delayed
import pyxit
from collections import defaultdict
from math import ceil
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from affine import Affine
from cytomine.models import Annotation, AnnotationCollection
from rasterio import features
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import box
from skimage import io
from scipy.ndimage import binary_erosion
from skimage import morphology
from skimage.color import rgb2gray, rgb2hsv
from sldc_cytomine.dump import dump_region

# size limit for a roi to avoid memory error (~ 5000 x 5000)
ROI_PIXEL_AREA_LIMIT = 2.5e7


class TermPriority(object):
  def __init__(self, terms):
    self._terms = terms
    self._priorities = {t: i + 1 for i, t in enumerate(terms)}
  
  def get_prio_term(self, terms):
    """Returns None if no valid term has been found."""
    if len(terms) == 0:
      return None
    order = [(self._priorities[t], t) for t in terms if t in self._priorities]
    if len(order) == 0:
      return None
    return sorted(order)[0][1]

  def __iter__(self):
    # iterate by reverse order of priority
    for term in self._terms[::-1]:
      yield term, self._priorities[term]
  
  @property
  def class_map(self):
    return self._priorities


def change_referential(p, offset=None, zoom_level=0, height=None):
  """
  Parameters
  ----------
  p: Polygon
    A shapely polygon
  offset: tuple
    (x, y) an offest to apply to the polygon
  height: int
    The height of the region containing the polygon (post-offset) 
  zoom_level: float
    Scale factor (exponent)
  """
  if offset is not None:
    p = affine_transform(p, [1, 0, 0, 1, -offset[0], -offset[1]])
  if height is not None:
    p = affine_transform(p, [1, 0, 0, -1, 0, height])
  p = affine_transform(p, [1 / 2 ** zoom_level, 0, 0, 1 / 2 ** zoom_level, 0, 0])
  return p


def extract_intersecting_by_term(roi: Annotation, others: AnnotationCollection, term_priority: TermPriority):
  """all Annotation must have a field 'polygon' with the annotation shapely geometry
  """
  intersecting = defaultdict(list)
  for other in others:
    if not roi.polygon.intersects(other.polygon) or roi.image != other.image:
      continue
    term = term_priority.get_prio_term(other.term)
    intersecting[term].append(other)
  return intersecting


def download_segmentation(
  id_project: int,
  id_roi_term: int,
  id_terms: list,
  working_path: str,
  id_terms_positive: list=None,
  zoom_level: int=0,
  id_users: list=None,
  id_images: list=None,
  n_jobs: int=0
):
  """Build locally a segmentation dataset from a cytomine project.

  Input images are defined with ROI annotations identified by a term identifier (`id_roi_term`).

  Foreground is represented by annotations identified by identifiers. Two cases:
  a) Multi-class segmentation: each term corresponds to a class. All areas not covered by an 
    annotation are associated with an additional 'background' class. The order of terms is 
    important as it defines the precedence in case of annotation overlap (i.e. a pixel covered by 
    several annotations will be associated the term of the annotation such that this term is the 
    first to appear in the list `id_terms`)
  b) Binary segmentation (when `id_terms_positive` is provided): all terms listed in `id_positive_terms`
    will be associated to the positive class. All others are relegated to the background. In case of 
    annotation overlap, positive terms have the precedence over negative/background terms.

  Resulting crops and masks files are written as '.png' into `crops_path` and `masks_path` respectively.
  Files are named '{ROI_IDENTIFIER}.png' in both directories.

  Parameters
  ----------
  id_project: int
    Identifier of the project
  id_roi_term: int
    Identifier of the region of interest term
  id_terms: list
    Identifier of terms
  working_path: str
    Path where to write the dataset.
  id_terms_positive: list
  zoom_level: int (default: 0)
    The zoom level (0 = highest resolution)
  id_users: list
    List of user identifiers to filter annotations (by annotators)
  id_images: list
    List of image identifiers to filter images (only take annotations and ROIs in these images)
  n_jobs: int
    To parallelize tile download when fetching the crop of a ROI

  Returns
  -------
  rois: AnnotationCollection
    Collection of ROI Annotations, each ROI should have a corresponding crop and mask in crops_path and masks_path
  crops_path: str
    Path to the crops
  masks_path: str
    Path to the masks.
  class_map: str
    Maps a term identifier with a class (1 -> n, 0 for background)
  is_binary: bool
    True if the dataset is binary, False otherwise
  """
  # annotation collection fetch common parameters
  fetch_params = {
    "showMeta": True, 
    "showWKT": True
  }
  if id_users is not None:
    fetch_params["users"] = id_users
  if id_images is not None:
    fetch_params["images"] = id_images

  # paths for downloaded data
  crops_path = os.path.join(working_path, "crops")
  os.makedirs(crops_path, exist_ok=True)
  masks_path = os.path.join(working_path, "masks")
  os.makedirs(masks_path, exist_ok=True)

  roi_collection = AnnotationCollection(**fetch_params, terms=[id_roi_term]).fetch_with_filter("project", id_project)

  # check all roi before processing
  for roi in roi_collection:
    roi.polygon = wkt.loads(roi.location)
    if box(*roi.polygon.bounds).area / 2 ** (2 * zoom_level) >= ROI_PIXEL_AREA_LIMIT:
      raise ValueError(f"""ROI from annotation #{roi.id} is too large, stopping processing to avoid out of memory error. 
      Crop area should not be larger than a ROI of 5000 x 5000 at the given zoom level (or of equivalent surface)""")

  if len(roi_collection) == 0:
    raise ValueError("no ROI selected")

  # exclude roi term from term list
  id_terms = [t for t in id_terms if t != id_roi_term]
  if id_terms_positive is not None:
    id_terms_positive = [t for t in id_terms_positive if t != id_roi_term]

  if len(id_terms) == 0:
    raise ValueError("no terms selected")
  if id_terms_positive is not None and len(id_terms_positive) == 0:
    id_terms_positive = None  # go back to multi-class

  terms_collection = AnnotationCollection(**fetch_params, terms=id_terms).fetch_with_filter("project", id_project)

  # build polygon once 
  for annotation in terms_collection:
    annotation.polygon = wkt.loads(annotation.location)

  # determine term priority to print crop
  if id_terms_positive:
    terms_by_priority = id_terms_positive + [t for t in id_terms if t not in id_terms_positive]
  else:
    terms_by_priority = id_terms
  term_priority = TermPriority(terms_by_priority)

  # to detect if a term was unseen during mask building  
  not_found_terms = set(term_priority.class_map.keys())

  for roi in roi_collection:
    # download crop
    with TemporaryDirectory() as tmpdir:
      roi.crop_path = os.path.join(crops_path, f"{roi.id}.png")
      dump_region(roi, roi.crop_path, zoom_level=zoom_level, n_jobs=n_jobs, working_path=tmpdir)

    intersecting = extract_intersecting_by_term(roi, terms_collection, term_priority)

    minx, miny, maxx, maxy = change_referential(roi.polygon, zoom_level=zoom_level).bounds
    height, width = ceil(maxy - int(miny)), ceil(maxx - int(minx))
    roi_affine = Affine.identity()
    class_mask = np.zeros([height, width], dtype=np.uint8)
    
    # this iterates from lowest priority to highest priority
    for term, class_value in term_priority:
      if id_terms_positive is not None and term not in id_terms_positive:
        not_found_terms.discard(term)
        continue  # don't draw background in case of binary
      roi_polygons = [a.polygon for a in intersecting[term]]
      if len(roi_polygons) == 0:
        continue
      term_mask = features.geometry_mask(
        geometries=[  # minx & maxx are post-zoom coordinates 
          change_referential(change_referential(p, zoom_level=zoom_level), offset=(minx, miny)) 
          for p in roi_polygons
        ],
        out_shape=class_mask.shape,
        transform=roi_affine,
        invert=True)
      class_mask[term_mask] = class_value if id_terms_positive is None else 1
      if np.any(term_mask):
        not_found_terms.discard(term) 

    mask_path = os.path.join(masks_path, f"{roi.id}.png")
    io.imsave(mask_path, class_mask[::-1], check_contrast=False)

  if len(not_found_terms) > 0:
    raise ValueError(f"some terms were not found during dataset building {not_found_terms}, be sure to provide examples for all requested terms or unselect the missing terms")

  class_map = term_priority.class_map
  if id_terms_positive is not None:
    class_map = {t: (1 if t in id_terms_positive else 0) for t in term_priority.class_map.keys()}

  return roi_collection, crops_path, masks_path, class_map, id_terms_positive is not None


def change_colorspace(rgb_img: np.ndarray, colorspace: int):
  if colorspace == pyxit.estimator.COLORSPACE_GRAY:
    return rgb2gray(rgb_img)
  elif colorspace == pyxit.estimator.COLORSPACE_RGB:
    return rgb_img
  elif colorspace == pyxit.estimator.COLORSPACE_HSV:
    return rgb2hsv(rgb_img)
  else:
    raise ValueError("unsupported color space use HSV, Gray or RGB")


def _subwindow(img: np.ndarray, mask: np.ndarray, candidate: np.ndarray, target_size: int=16):
  y, x = candidate
  y_end, x_end = y + target_size, x + target_size
  if y_end > img.shape[0] or x_end > img.shape[1]:
    raise ValueError(f"window out the image: {img.shape}, coords: y:{y} x:{x} (size: {target_size})") 
  if y_end > mask.shape[0] or x_end > mask.shape[1]:
    raise ValueError(f"window out the mask: {mask.shape}, coords: y:{y} x:{x} (size: {target_size})") 
  return img[y:y_end, x:x_end].flatten(), mask[y:y_end, x:x_end].flatten()


def extract_subwindows(
  rois: AnnotationCollection,
  crops_path: str,
  masks_path: str,
  total_n_subwindows: int,
  target_size: int=16,
  colorspace: int=pyxit.estimator.COLORSPACE_HSV,
  zoom_level: int=0,
  n_jobs=0,
  rng: np.random.Generator=None
):
  """Extract subwindows that fit inside the ROIs 

  Parameters
  ----------
  rois: AnnotationCollection
    Collection of roi annotations (there should be one crop and mask in crops_path and masks_path
    for each roi in this collection, their name should be ROI_ID.png)
  crops_path: str
    Path to crop files (corresponding mask filename should match filename in this folder )
  masks_path: str
    Path to mask files
  total_n_subwindows: int
    Total number of subwindows to extract
  target_size: int
    Windows target width and height in pixels
  interpolation: int
    see pyxit.estimators.INTERPOLATION_* variables
  colorspace: int
    see pyxit.estimators.COLORSPACE_* variables
  zoom_level: int (default: 0)
    The zoom level (0 = highest resolution)
  n_jobs: int
    Number of jobs to use for fetching the tiles
  rng: np.random.Generator
    A random number generator. Default: generated from default_rng 
  """
  if rng is None:
    rng = np.random.default_rng()

  # compute candidate windows for all ROI (all pixels must fit within the ROI)
  roi_candidates = dict()
  window_footprint = morphology.square(target_size, dtype=bool)

  for roi in rois:
    original_polygon = wkt.loads(roi.location)
    o_minx, o_miny, _, o_maxy = original_polygon.bounds
    roi_polygon = change_referential(
      original_polygon, 
      offset=(int(o_minx), int(o_miny)), 
      zoom_level=zoom_level, 
      height=ceil(o_maxy - o_miny)
    )

    _, _, maxx, maxy = roi_polygon.bounds
    roi_mask = features.geometry_mask(
      geometries=[roi_polygon], 
      out_shape=[ceil(maxy), ceil(maxx)], 
      transform=Affine.identity(),
      all_touched=True, invert=True).astype(bool)
    # binary_erosion => resulting mask contains 1 for all pixels where a window fits in the ROI
    candidates_mask = binary_erosion(roi_mask, structure=window_footprint, origin=-(target_size // 2))
    roi_candidates[roi.id] = np.vstack(np.where(candidates_mask)).transpose()

  total_candidates = sum([candidates.shape[0] for candidates in roi_candidates.values()])

  if total_candidates < total_n_subwindows:
    warnings.warn(f"number of subwindows to extract ({total_n_subwindows}) seems larger than the number of candidates in the ROIs ({total_candidates}). Is it ok ?")
  
  subwindows, masks = list(), list() 
  with Parallel(n_jobs=n_jobs) as parallel:
    for roi in rois:
      candidates = roi_candidates[roi.id]
      n_subwindows = round(total_n_subwindows * candidates.shape[0] / total_candidates)
      filename = f"{roi.id}.png"

      # load images
      crop_filepath = os.path.join(crops_path, filename)
      crop_image = io.imread(crop_filepath)
      crop_image = change_colorspace(crop_image, colorspace)
      mask_filepath = os.path.join(masks_path, filename)
      mask_image = io.imread(mask_filepath)
      
      windows_coords = rng.choice(candidates, size=n_subwindows, replace=True, axis=0)
      output = parallel(delayed(_subwindow)(crop_image, mask_image, w_coords, target_size) for w_coords in windows_coords)
      roi_crops, roi_masks = zip(*output)
      subwindows.extend(roi_crops)
      masks.extend(roi_masks)
  
  return np.vstack(subwindows), np.vstack(masks)