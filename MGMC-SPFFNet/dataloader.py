import ABIDEParser as Reader
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 1500
        self.num_classes = 2
    def load_data(self, connectivity='correlation', atlas1='aal',atlas2='ho',atlas3='cc200'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)#

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX')
        values_list = [float(value) for value in ages.values()]

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int64)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int64)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
        self.y = y - 1
        #PC
        self.raw_features1 = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas1)
        self.raw_features2 = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas2)
        self.raw_features3 = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas3)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age

        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:, 2])
        return self.raw_features1,self.raw_features2,self.raw_features3, self.y, phonetic_data,
    def data_3split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state = 42)
        all_raw_features = np.concatenate((self.raw_features1, self.raw_features2, self.raw_features3), axis=1)
        cv_splits = list(skf.split(all_raw_features, self.y))
        return cv_splits

    def get_node_edge_features(self,nodeftr_dim, train_ind):
        '''preprocess node features for gcn
        '''
        self.node_ftr_dim = nodeftr_dim
        self.node_ftr1 = Reader.feature_selection(self.raw_features1, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr2 = Reader.feature_selection(self.raw_features2, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr3 = Reader.feature_selection(self.raw_features3, self.y, train_ind, self.node_ftr_dim)

        self.node_ftr1 = preprocess_features(self.node_ftr1)  # D^-1 dot node_ftr
        self.node_ftr2 = preprocess_features(self.node_ftr2)  # D^-1 dot node_ftr
        self.node_ftr3 = preprocess_features(self.node_ftr3)

        self.edge_ftr1 = self.node_ftr1
        self.edge_ftr2 = self.node_ftr2
        self.edge_ftr3 = self.node_ftr3

        return self.node_ftr1,self.node_ftr2,self.node_ftr3,self.edge_ftr1,self.edge_ftr2,self.edge_ftr3

    def get_AELN_inputs(self, nodeftr):
        # construct edge network inputs
        n = self.node_ftr1.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = nodeftr.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)

        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(nodeftr, self.pd_dict)

        flatten_ind = 0

        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nodeftr[i], nodeftr[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.5)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]
        return edge_index, edgenet_input


if __name__ == "__main__":
    site = np.zeros([4], dtype=np.int)
    print(site)
    print(site.shape)
