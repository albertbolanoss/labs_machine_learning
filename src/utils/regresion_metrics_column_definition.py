import numpy as np

class ColumnInfo:
    def __init__(self, name, desc, dtype, format_str=None):
        """
        Initializes the feature information.
        
        Args:
            name (str): The name of the column.
            desc (str): The description of the feature.
            dtype (type): The expected data type (e.g., np.int64, str).
            format_str (str, optional): The format string for display.
        """
        self.name = name
        self.desc = desc
        self.dtype = dtype
        self.format_str = format_str

class Metric:
    MEAN_TEST_NEG_RMSE = ColumnInfo("mean_test_neg_rmse", "Mean Test Negative RMSE.", np.int64, format_str='$ {:,.4f}')
    MEAN_TEST_R2 = ColumnInfo("mean_test_r2", "Mean Test R2.", np.int64, format_str='$ {:,.4f}')
    PCA_COMPONENTS = ColumnInfo("param_pca__n_components", "PCA Components.", np.int64, format_str='$ {:,.0f}')
    SCORE = ColumnInfo("score", "Final Score.", np.int64, format_str='$ {:,.8f}')