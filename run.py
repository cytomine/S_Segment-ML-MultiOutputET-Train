import os

import joblib
import numpy as np
from pathlib import Path

from cytomine.models import AttachedFile, Property, Job
from pyxit import build_models
from pyxit.estimator import _get_output_from_mask
from cytomine import CytomineJob
from cytomine.utilities.software import parse_domain_list, setup_classify, stringify


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # use only images from the current project
        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")

        # hardcode parameter for setup classify to fetch alphamask instead of plain crop.
        cj.parameters.cytomine_download_alpha = True
        cj.parameters.cytomine_id_projects = "{}".format(cj.project.id)
        print(cj.parameters)
        base_path, downloaded = setup_classify(
            args=cj.parameters, logger=cj.job_logger(1, 40),
            dest_pattern=os.path.join("{term}", "{image}_{id}.png"),
            root_path=Path.home(), set_folder="train", showTerm=True
        )

        x = np.array([f for annotation in downloaded for f in annotation.filenames])
        y = np.array([int(os.path.basename(os.path.dirname(filepath))) for filepath in x])

        # transform classes
        cj.job.update(progress=50, statusComment="Transform classes...")
        classes = parse_domain_list(cj.parameters.cytomine_id_terms)
        positive_classes = parse_domain_list(cj.parameters.cytomine_positive_terms)
        classes = np.array(classes) if len(classes) > 0 else np.unique(y)
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

        if cj.parameters.cytomine_binary:
            cj.logger.info("Will be training on 2 classes ({} classes before binarization).".format(n_classes))
            y = np.in1d(y, positive_classes).astype(np.int)
        else:
            cj.logger.info("Will be training on {} classes.".format(n_classes))
            y = np.searchsorted(classes, y)

        # build model
        cj.job.update(progress=55, statusComment="Build model...")
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
            create_svm=cj.parameters.svm,
            C=cj.parameters.svm_c,
            random_state=cj.parameters.seed,
            n_estimators=cj.parameters.forest_n_estimators,
            min_samples_split=cj.parameters.forest_min_samples_split,
            max_features=cj.parameters.forest_max_features,
            n_jobs=cj.parameters.n_jobs
        )

        # to extract the classes form the mask
        pyxit.get_output = _get_output_from_mask

        cj.job.update(progress=60, statusComment="Train model...")
        pyxit.fit(x, y)

        cj.job.update(progress=90, statusComment="Save model....")
        model_filename = joblib.dump(pyxit, os.path.join(base_path, "model.joblib"), compress=3)[0]

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_filename,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        Property(cj.job, key="classes", value=stringify(classes)).save()
        Property(cj.job, key="binary", value=cj.parameters.cytomine_binary).save()
        Property(cj.job, key="positive_classes", value=stringify(positive_classes)).save()

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
