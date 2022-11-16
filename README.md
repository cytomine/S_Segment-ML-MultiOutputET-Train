# Cytomine Software - Segment-ML-MultiOutputET-Train-ROI

> This repository is a fork of an app developped by ULiège Cytomine Research team: [Cytomine-ULiege/S_Segment-ML-MultiOutputET-Train](https://github.com/Cytomine-ULiege/S_Segment-ML-MultiOutputET-Train)

A Cytomine (https://cytomine.org) app for segmentation of samples using a segmentation (ML) model. This differs from the original ULiège implementation by the fact that the training process is performed on annotated regions of interests rather than crops of annotations. 

For the associated prediction software, see:
- [cytomine/S_Segment-ML-MultiOutputET-Pred-BI](https://github.com/cytomine/S_Segment-ML-MultiOutputET-Pred-BI)
- [cytomine/S_Segment-ML-MultiOutputET-Pred-ROI](https://github.com/cytomine/S_Segment-ML-MultiOutputET-Pred-ROI)

This implementation follows Cytomine (> v3.0) external app conventions based on container technology.

**Summary:** It trains a segmentation model based on random subwindows and multiple output extra-trees.

**Typical application:** Predict the regions of interest from chosen areas that corresponds to a certain term (*e.g.* tumor regions in histology slides).

**Based on:** pixel classification model to detect regions of interest, the methodology is presented [here](https://ieeexplore.ieee.org/document/6868017).

## Parameters

### Connection to Cytomine

**Parameters**: *cytomine_host*, *cytomine_public_key*, *cytomine_private_key*, *cytomine_id_project* and *cytomine_id_software* 
 
These are parameters needed for Cytomine external app. They will allow the app to be run on the Cytomine instance (determined with its host), connect and communicate with the instance (with the key pair). An app is always run within a project (*cytomine_id_project*) and to be run, the app must be previously declared to the plateform as a software (*cytomine_id_software*). 

### Data selection

**Parameters**: *cytomine_id_images*, *cytomine_id_roi_term*, *cytomine_zoom_level*

The algorithm will only use the images listed for training (*cytomine_id_images*), or all the images of the project if the parameter is omitted or empty. The images will be opened at the provided zoom level (*cytomine_zoom_level*, 0 for highest magnification). In the selected images, only pixels in the provided region of interests (ROI) will be considered by the training algorithms. The ROI are determined by a term identifier (*cytomine_id_roi_term*).

### Learning problem

**Parameters**: *cytomine_id_terms*, *cytomine_id_positive_terms*, *cytomine_id_users*

The app uses annotations to identify areas to segment and train the algorithm. Moreover, only annotations contained in the provided regions of interests and made by the listed users (*cytomine_id_users*, all users are considered if this parameter is empty or omitted) will be used.

The segmentation problem can be **multi-class** (*e.g.* tumor, non-tumor and undetermined) in which case the classes must be provided with a list of term identifiers (one for each class, *cytomine_id_terms*) that have been associated with the corresponding annotations. All areas not covered by an annotation will be assigned to an additional 'background' class.

The segmentation problem can be **binary** (*e.g.* tumor and non-tumor, tissue and background) in which case term identifiers must be provided for all terms of interest (positive and negative, *cytomine_id_terms*). Moreover, identifiers terms considered to represent the positive class must be provided (*cytomine_id_positive_terms*). All terms listed as positive will be assigned to the 'foreground' class (value 1), all the other terms listed as well as areas not covered by any annotation will be assigned to the 'background' class (value 0).

### Learning algorithm

**Parameters**: *pyxit_n_total_subwindows*, *forest_n_estimators*, *pyxit_target_size*, *pyxit_colorspace*, *forest_min_samples_split*, *forest_max_features*, *random_seed*

These parameters can be tuned to modify the behaviour of the underlying algorithms and (hopefully) improve performance of the generated model. Beware that some parameters can have a significant impact to the running time and required memory of the app. The default values should provide a good baseline with acceptable running time and memory requirements. 

The total number of subwindows (*pyxit_n_total_subwindows*) and number of estimators (*forest_n_estimators*) can be increased for better performance.

### Execution 

**Parameters**: *n_jobs*

Number of CPUs to use for executing the computations (*n_jobs*, -1 for all CPUs).

## Output

The software produces three outputs all attached to the Cytomine Job:

1. attached file `model.joblib`: the resulting model file
2. property `binary`: true if the model is binary, false if it is multi-class
3. property `classes`: a comma-separated list of classes. For a binary problem, always `0,1`. For a multi-class problem, `classes[i] (i > 0)` is the term corresponding to class `i` predicted by `model.joblib` (and `classes[0]` is the background, hence not a term, and is always `0`)  



