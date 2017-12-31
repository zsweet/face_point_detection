# file kfkd.py

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from nolearn.lasagne import BatchIterator


class Load():
    def __init__(self):
        self.FTRAIN = 'training.csv'
        self.FTEST = 'test.csv'

    def load(self,test=False, cols=None):
        """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
        Pass a list of *cols* if you're only interested in a subset of the
        target columns.
        """
        fname = self.FTEST if test else self.FTRAIN
        df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

        # The Image column has pixel values separated by space; convert
        # the values to numpy arrays:
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)

        if not test:  # only FTRAIN has any target columns
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48  # scale target coordinates to [-1, 1]
            X, y = shuffle(X, y, random_state=42)  # shuffle train data
            y = y.astype(np.float32)
        else:
            y = None

        return X, y

    def load2d(self,test=False,cols=None):
        X,y = self.load(test=test)
        X = X.reshape(-1,1,96,96)
        return X,y




class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

class Specialist_settings():
    def __init__(self):
        SPECIALIST_SETTINGS = [
            dict(
                columns=(
                    'left_eye_center_x', 'left_eye_center_y',
                    'right_eye_center_x', 'right_eye_center_y',
                ),
                flip_indices=((0, 2), (1, 3)),
            ),

            dict(
                columns=(
                    'nose_tip_x', 'nose_tip_y',
                ),
                flip_indices=(),
            ),

            dict(
                columns=(
                    'mouth_left_corner_x', 'mouth_left_corner_y',
                    'mouth_right_corner_x', 'mouth_right_corner_y',
                    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
                ),
                flip_indices=((0, 2), (1, 3)),
            ),

            dict(
                columns=(
                    'mouth_center_bottom_lip_x',
                    'mouth_center_bottom_lip_y',
                ),
                flip_indices=(),
            ),

            dict(
                columns=(
                    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
                    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
                ),
                flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
            ),

            dict(
                columns=(
                    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
                    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
                ),
                flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
            ),
        ]

class PointName():
    def __init__(self):
        self.point = {'left_eye_center_x':0,
                      'left_eye_center_y':1,
                      'right_eye_center_x':2,
                      'right_eye_center_y': 3,
                      'left_eye_inner_corner_x':4,
                      'left_eye_inner_corner_y': 5,
                      'left_eye_outer_corner_x':6,
                      'left_eye_outer_corner_y': 7,
                      'right_eye_inner_corner_x':8,
                      'right_eye_inner_corner_y': 9,
                      'right_eye_outer_corner_x':10,
                      'right_eye_outer_corner_y': 11,
                      'left_eyebrow_inner_end_x':12,
                      'left_eyebrow_inner_end_y': 13,
                      'left_eyebrow_outer_end_x':14,
                      'left_eyebrow_outer_end_y': 15,
                      'right_eyebrow_inner_end_x':16,
                      'right_eyebrow_inner_end_y': 17,
                      'right_eyebrow_outer_end_x': 18,
                      'right_eyebrow_outer_end_y': 19,
                      'nose_tip_x':20,
                      'nose_tip_y': 21,
                      'mouth_left_corner_x':22,
                      'mouth_left_corner_y': 23,
                      'mouth_right_corner_x':24,
                      'mouth_right_corner_y': 25,
                      'mouth_center_top_lip_x':26,
                      'mouth_center_top_lip_y': 27,
                      'mouth_center_bottom_lip_x':28,
                      'mouth_center_bottom_lip_y': 29
                      }

if __name__ == '__main__':
    tmpload = Load()
    X, y = tmpload.load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))