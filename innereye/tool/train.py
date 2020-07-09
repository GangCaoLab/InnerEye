import json
import typing as t
import pickle

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from pathos.multiprocessing import ProcessingPool as Pool

from .base import ChainTool, ImgIO, SpotsIO
from ..lib.log import print_arguments
from ..lib.img.misc import get_img_2d, extract_sub_2d

from logging import getLogger
log = getLogger(__file__)


class SpotsClfModel2d(ChainTool, ImgIO, SpotsIO):

    def __init__(self, n_workers=1):
        self.anns: t.List[t.Dict] = []
        self.sample = None
        self.cycles = None
        self.model = None
        self._need_flatten = None
        self.spots = None
        self.n_workers = n_workers
        self._width = None
        self._border_default = None

    def load_annotation(self, path: str):
        """ Load annotations.

        :param path: Path to annotation json file.
        :return:
        """
        print_arguments(log.info)
        with open(path) as f:
            self.anns += json.load(f)
        return self

    def compose_sample(self, width=3, border_default=0):
        """ Compose samples for training.

        :param width: pixels width around center.
        :param border_default: default value of border.
        :return:
        """
        print_arguments(log.info)
        self._width = width
        self._border_default = border_default
        Xs = []
        ys = []
        for ann in self.anns:
            pts_pos = ann['points'].get('pos', [])
            pts_neg = ann['points'].get('neg', [])
            if (pts_pos == []) and (pts_neg == []):
                continue
            img_path = ann['img_path']
            self.read_img(img_path)
            im4d = self.cycles[ann['cycle'][0]]
            im2d = get_img_2d(im4d, ann['channel'], ann['z'])
            points = np.array(pts_pos + pts_neg)
            X = np.stack(extract_sub_2d(im2d, points, width, border_default))
            y = np.r_[np.ones(len(pts_pos)), np.zeros(len(pts_neg))]
            Xs.append(X)
            ys.append(y)
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        self.sample = X, y
        log.info(f"X shape: {X.shape}")
        log.info(f"{X.shape[0]} samples, {y[y==1].shape[0]} positive, {y[y==0].shape[0]} negative.")
        return self

    def _train(self, X, y, model, cv):
        if cv:
            scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
            scores = cross_validate(model, X, y, scoring=scoring, cv=cv)
            msg = f"Cross validation(k={cv}):\n"
            for k in scoring:
                score_ = scores['test_'+k]
                msg += f"{k}: mean: {np.mean(score_)}, std: {np.std(score_)}\n"
            log.info(msg)
        log.info("Fit with all samples.")
        model.fit(X, y)
        self.model = model

    def train_rf(self, model_args={'n_estimators': 100}, cv=5):
        """ Train Random Forest Classifier.

        :param model_args:
        :param cv:
        :return:
        """
        print_arguments(log.info)
        model = RandomForestClassifier(**model_args)
        X, y = self.sample
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        self._need_flatten = True
        self._train(X, y, model, cv)
        return self

    def save_model(self, path):
        """ Save classification model to file.

        :param path: Target file path.
        :return:
        """
        print_arguments(log.info)
        with open(path, 'wb') as f:
            pickle.dump((self.model, self._width, self._border_default), f)
        return self

    def load_model(self, path):
        """ Load classification from file.

        :param path:
        :return:
        """
        print_arguments(log.info)
        with open(path, 'rb') as f:
            self.model, self._width, self._border_default = pickle.load(f)
            if type(self.model) is RandomForestClassifier:
                self._need_flatten = True
        return self

    def filter_spots(self):
        print_arguments(log.info)
        pool = Pool(ncpus=self.n_workers)
        map_ = map if self.n_workers <= 1 else pool.imap
        idx = [(ixcy, ixch)
               for ixcy in range(len(self.spots))
               for ixch in range(len(self.spots[ixcy]))]

        def proc(ix_t):
            ixcy, ixch = ix_t
            spts = self.spots[ixcy][ixch]
            im4d = self.cycles[ixcy]
            spts_new = []
            for z in np.unique(spts[:, 2]):
                pts = spts[spts[:, 2] == z][:, :2]
                im2d = get_img_2d(im4d, ixch, int(z))
                subs = extract_sub_2d(im2d, np.c_[pts[:,1], pts[:,0]],
                                      self._width, self._border_default)
                x = np.stack(subs)
                if self._need_flatten:
                    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
                y = self.model.predict(x)
                pts_f = pts[y == 1]
                pts_new = np.c_[pts_f, np.full(pts_f.shape[0], z)]
                spts_new.append(pts_new)
            spts_new = np.concatenate(spts_new)
            return ix_t, spts_new

        spots = [[] for _ in range(len(self.spots))]
        for (ixcy, ixch), spts in map_(proc, idx):
            spots[ixcy].append(spts)
        self.spots = spots
        return self


if __name__ == "__main__":
    from ..lib.log import set_global_logging
    set_global_logging()
    import fire
    fire.Fire(SpotsClfModel2d)
