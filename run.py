import os

import joblib
import numpy as np

from cytomine.models import AttachedFile, Property, Job
from pyxit import build_models
from pyxit.estimator import _get_output_from_mask
from cytomine import CytomineJob
from cytomine.utilities.software import parse_domain_list, setup_classify, stringify


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # use only images from the current project
        cj.job.update(progress=1, statusComment="Preparing execution (creating folders,...).")

        # hardcode parameter for setup classify to fetch alphamask instead of plain crop.
        cj.parameters.cytomine_download_alpha = True
        cj.parameters.cytomine_id_projects = "{}".format(cj.project.id)
        cj.job.update(progress=2, statusComment="Downloading crops.")
        base_path, downloaded = setup_classify(
            args=cj.parameters, logger=cj.job_logger(2, 25),
            dest_pattern=os.path.join("{term}", "{image}_{id}.png"),
            root_path=str("tmp"), set_folder="train", showTerm=True
        )

        x = np.array([f for annotation in downloaded for f in annotation.filenames])
        y = np.array([int(os.path.basename(os.path.dirname(filepath))) for filepath in x])

        # transform classes
        cj.job.update(progress=25, statusComment="Transform classes...")
        positive_terms = parse_domain_list(cj.parameters.cytomine_id_positive_terms)
        selected_terms = parse_domain_list(cj.parameters.cytomine_id_terms)
        is_binary = len(selected_terms) > 0 and len(positive_terms) > 0
        foreground_terms = np.unique(y) if len(selected_terms) == 0 else np.array(selected_terms)
        if len(positive_terms) == 0:
            classes = np.hstack((np.zeros((1,), dtype=int), foreground_terms))
        else:  # binary
            foreground_terms = np.array(positive_terms)
            classes = np.array([0, 1])
            # cast to binary
            fg_idx = np.in1d(y, list(foreground_terms))
            bg_idx = np.in1d(y, list(set(selected_terms).difference(foreground_terms)))
            y[fg_idx] = 1
            y[bg_idx] = 0

        n_classes = classes.shape[0]

        # filter unwanted terms
        cj.logger.info("Size before filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))
        keep = np.in1d(y, classes)
        x, y = x[keep], y[keep]
        cj.logger.info("Size after filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))

        if x.shape[0] == 0:
            raise ValueError("No training data")

        if is_binary:
            # 0 (background) vs 1 (classes in foreground )
            cj.logger.info("Binary segmentation:")
            cj.logger.info("> class '0': background & terms {}".format(set(selected_terms).difference(positive_terms)))
            cj.logger.info("> class '1': {}".format(set(foreground_terms)))
        else:
            # 0 (background vs 1 vs 2 vs ... n (n classes from cytomine_id_terms)
            cj.logger.info("Multi-class segmentation:")
            cj.logger.info("> background class '0'")
            cj.logger.info("> term classes: {}".format(set(foreground_terms)))

        # build model
        cj.job.update(progress=27, statusComment="Build model...")
        et, pyxit = build_models(
            n_subwindows=cj.parameters.pyxit_n_subwindows,
            min_size=cj.parameters.pyxit_min_size,
            max_size=cj.parameters.pyxit_max_size,
            target_width=cj.parameters.pyxit_target_width,
            target_height=cj.parameters.pyxit_target_height,
            interpolation=cj.parameters.pyxit_interpolation,
            transpose=cj.parameters.pyxit_transpose,
            colorspace=cj.parameters.pyxit_colorspace,
            fixed_size=cj.parameters.pyxit_fixed_size,
            verbose=int(cj.logger.level == 10),
            random_state=cj.parameters.seed,
            n_estimators=cj.parameters.forest_n_estimators,
            min_samples_split=cj.parameters.forest_min_samples_split,
            max_features=cj.parameters.forest_max_features,
            n_jobs=cj.parameters.n_jobs
        )

        # to extract the classes form the mask
        pyxit.get_output = _get_output_from_mask

        # extract subwindows manually to avoid class problem
        cj.job.update(progress=30, statusComment="Extract subwindwos...")
        _x, _y = pyxit.extract_subwindows(x, y)

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
        model_filename = joblib.dump(pyxit, os.path.join(base_path, "model.joblib"), compress=3)[0]

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
