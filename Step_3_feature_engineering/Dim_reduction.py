from sklearn.decomposition import PCA, FastICA
import util.util as util

class PrincipalComponentAnalysis:
    def __init__(self):
        self.pca = []

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)
        # perform the PCA.
        self.pca = PCA(n_components = len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components = number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        #And add the new ones:
        for comp in range(0, number_comp):
            data_table['pca_' +str(comp+1)] = new_values[:,comp]

        return data_table

class IndependentComponentAnalysis:
    def __init__(self):
        self.ica = []

    # Apply a FastICA given the number of components we have selected.
    # We add new pca columns.
    def apply_ica(self, data_table, cols):
        # Normalize the data first.
        dt_norm = util.normalize_dataset(data_table, cols)

        # perform the FastICA for all components.
        self.ica = FastICA(n_components = len(cols))
        self.ica.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.ica.transform(dt_norm[cols])

        #And add the new ones:
        for comp in range(0, len(cols)):
            data_table['FastICA_' +str(comp+1)] = new_values[:,comp]

        return data_table