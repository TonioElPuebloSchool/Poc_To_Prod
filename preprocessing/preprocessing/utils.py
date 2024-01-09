import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

def integer_floor(float_value: float):
    return int(np.floor(float_value))

class _SimpleSequence(Sequence):
    def __init__(self, get_batch_method, num_batches_method):
        self.get_batch_method = get_batch_method
        self.num_batches_method = num_batches_method

    def __len__(self):
        return self.num_batches_method()

    def __getitem__(self, idx):
        return self.get_batch_method()

class BaseTextCategorizationDataset:
    def __init__(self, batch_size, train_ratio=0.8):
        assert train_ratio < 1.0
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def _get_label_list(self):

        raise NotImplementedError

    def get_num_labels(self):
        # TODO: CODE HERE
        return len(self._get_label_list())

    def _get_num_samples(self):
        raise NotImplementedError

    def _get_num_train_samples(self):
        # TODO: CODE HERE
        return integer_floor(self.train_ratio * self._get_num_samples())

    def _get_num_test_samples(self):
        # TODO: CODE HERE
        return self._get_num_samples() - self._get_num_train_samples()

    def _get_num_train_batches(self):
        # TODO: CODE HERE
        return integer_floor(self._get_num_train_samples() / self.batch_size)

    def _get_num_test_batches(self):
        # TODO: CODE HERE
        return integer_floor(self._get_num_test_samples() / self.batch_size)

    def get_train_batch(self):
        raise NotImplementedError

    def get_test_batch(self):
        raise NotImplementedError

    def get_index_to_label_map(self):
        # TODO: CODE HERE
        return {i: label for i, label in enumerate(self._get_label_list())}

    def get_label_to_index_map(self):
        # TODO: CODE HERE
        return {label: i for i, label in enumerate(self._get_label_list())}

    def to_indexes(self, labels):
        # TODO: CODE HERE
        label_to_index_map = self.get_label_to_index_map()
        return [label_to_index_map[label] for label in labels]

    def get_train_sequence(self):
        return _SimpleSequence(self.get_train_batch, self._get_num_train_batches)

    def get_test_sequence(self):
        # TODO: CODE HERE
        return _SimpleSequence(self.get_test_batch, self._get_num_test_batches)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(n_train_samples: {self._get_num_train_samples()}, "
            f"n_test_samples: {self._get_num_test_samples()}, "
            f"n_labels: {self.get_num_labels()})"
        )

class LocalTextCategorizationDataset(BaseTextCategorizationDataset):
    def __init__(
        self, filename, batch_size, train_ratio=0.8, min_samples_per_label=100, preprocess_text=lambda x: x
    ):
        super().__init__(batch_size, train_ratio)
        self.filename = filename
        self.preprocess_text = preprocess_text

        self._dataset = self.load_dataset(filename, min_samples_per_label)

        assert self._get_num_train_batches() > 0
        assert self._get_num_test_batches() > 0

        # TODO: CODE HERE
        # from self._dataset, compute the label list
        self._label_list = self._get_label_list()

        y = self.to_indexes(self._dataset["tag_name"])
        y = to_categorical(y, num_classes=len(self._label_list))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._dataset["title"],
            y,
            train_size=self._get_num_train_samples(),
            stratify=y,
        )

        self.train_batch_index = 0
        self.test_batch_index = 0

    @staticmethod
    def load_dataset(filename, min_samples_per_label):
        # TODO: CODE HERE
        dataset = pd.read_csv(filename)
        # assert that columns are the ones expected
        # TODO: CODE HERE
        def filter_tag_position(position):
            def filter_function(df):
                """
                keep only tag_position = position
                """
                # TODO: CODE HERE
                return df[df['tag_position'] == position]

            return filter_function

        def filter_tags_with_less_than_x_samples(min_samples):
            def filter_function(df):
                # TODO: CODE HERE
                return df.groupby('tag_name').filter(lambda x: len(x) >= min_samples)

            return filter_function
        return (
            dataset.pipe(filter_tag_position(0))
            .pipe(filter_tags_with_less_than_x_samples(min_samples_per_label))
            .reset_index(drop=True)
        )

    def _get_label_list(self):
        # TODO: CODE HERE
        return self._dataset["tag_name"].unique().tolist()

    def _get_num_samples(self):
        # TODO: CODE HERE
        return len(self._dataset)

    def get_train_batch(self):
        i = self.train_batch_index
        # TODO: CODE HERE
        next_x = self.preprocess_text(
            self.x_train[i * self.batch_size : (i + 1) * self.batch_size]
        )
        next_y = self.y_train[i * self.batch_size : (i + 1) * self.batch_size]
        # When we reach the max num batches, we start anew
        self.train_batch_index = (self.train_batch_index + 1) % self._get_num_train_batches()
        return next_x, next_y

    def get_test_batch(self):
        # TODO: CODE HERE
        i = self.test_batch_index
        next_x = self.preprocess_text(
            self.x_test[i * self.batch_size : (i + 1) * self.batch_size]
        )
        next_y = self.y_test[i * self.batch_size : (i + 1) * self.batch_size]
        self.test_batch_index = (self.test_batch_index + 1) % self._get_num_test_batches()
        return next_x, next_y
