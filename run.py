import os
from pathlib import Path

import joblib
import numpy as np
from cytomine import CytomineJob
from cytomine.models import AttachedFile, Job, Property
from cytomine.utilities.software import parse_domain_list, stringify
from pyxit import build_models

from dataset import download_segmentation, extract_subwindows


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # use only images from the current project
        cj.job.update(progress=1, statusComment="Preparing execution (creating folders,...).")

        # hardcode parameter for setup classify to fetch alphamask instead of plain crop.
        cj.job.update(progress=2, statusComment="Downloading ROI crops and masks.")
        home = Path.home()
        data_path = os.path.join(home, "data")
        
        rois, crops_path, masks_path, class_map, is_binary = download_segmentation(
            id_project=cj.project.id,
            id_roi_term=int(cj.parameters.cytomine_id_roi_term),
            id_terms=parse_domain_list(cj.parameters.cytomine_id_terms),
            working_path=data_path,
            id_terms_positive=parse_domain_list(cj.parameters.cytomine_id_positive_terms),
            zoom_level=cj.parameters.cytomine_zoom_level,
            id_users=cj.parameters.cytomine_id_users,
            id_images=cj.parameters.cytomine_id_images,
            n_jobs=cj.parameters.n_jobs
        )

        if is_binary: 
            classes = np.array([0, 1])
        else:
            classes = np.array([0] + sorted(class_map.values()))
        n_classes = classes.shape[0]
        cj.job.update(progress=25, statusComment=f"Dataset built, found {len(rois)} ROIs, {'binary' if is_binary else 'multi-class'} segmentation ({len(classes)} classes)...")

        # build model
        _, pyxit = build_models(
            n_subwindows=1,   # subwindows generated manually below
            target_width=cj.parameters.pyxit_target_size,
            target_height=cj.parameters.pyxit_target_size,
            colorspace=cj.parameters.pyxit_colorspace,
            fixed_size=True,
            verbose=int(cj.logger.level == 10),
            random_state=cj.parameters.random_seed,
            n_estimators=cj.parameters.forest_n_estimators,
            min_samples_split=cj.parameters.forest_min_samples_split,
            max_features=cj.parameters.forest_max_features,
            n_jobs=cj.parameters.n_jobs
        )

        # extract subwindows manually to avoid class problem
        cj.job.update(progress=30, statusComment="Extract subwindows...")
        _x, _y = extract_subwindows(
            rois, 
            crops_path=crops_path,
            masks_path=masks_path,
            total_n_subwindows=cj.parameters.pyxit_n_total_subwindows,
            target_size=cj.parameters.pyxit_target_size,
            colorspace=cj.parameters.pyxit_colorspace,
            zoom_level=cj.parameters.cytomine_zoom_level,
            n_jobs=cj.parameters.n_jobs,
            rng=np.random.default_rng(cj.parameters.random_seed)
        )

        actual_classes = np.unique(_y)
        if actual_classes.shape[0] != classes.shape[0]:
            raise ValueError("Some classes are missing from the dataset: actual='{}', expected='{}'".format(
                ",".join(map(str, actual_classes)),
                ",".join(map(str, classes))
            ))

        cj.logger.info("Size of actual training data:")
        cj.logger.info(" - x   : {}".format(_x.shape))
        cj.logger.info(" - y   : {}".format(_y.shape))
        cj.logger.info(" - dist: {}".format(", ".join(["{}: {}".format(v, c) for v, c in zip(*np.unique(_y, return_counts=True))])))

        cj.job.update(progress=60, statusComment="Train model...")
        # "re-implement" pyxit.fit to avoid incorrect class handling
        pyxit.classes_ = classes
        pyxit.n_classes_ = n_classes
        pyxit.base_estimator.fit(_x, _y)

        cj.job.update(progress=90, statusComment="Save model....")
        model_filename = joblib.dump(pyxit, os.path.join(data_path, "model.joblib"), compress=3)[0]

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_filename,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        Property(cj.job, key="classes", value=stringify(classes)).save()
        Property(cj.job, key="binary", value=is_binary).save()

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
