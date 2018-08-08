import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import spearmanr

from bob_stim import get_stim  #, get_model_features, compute_similarity_matrices

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.models import Model

from convnetskeras.convnets import AlexNet


import sys
from stim_rep import get_image_stimulus

np.seterr('raise')

IMAGENET = True

def get_base_model(model_class):

    if IMAGENET:
        weights='imagenet'
    else:
        weights=None
    base_model = model_class(weights=weights, include_top=False)

    return base_model

def get_model_rep(model_class, layer_list=None):

    base_model = get_base_model(model_class)
    # print base_model.layers

    if layer_list is None:
        output_list = [layer.output for layer in base_model.layers]
    else:
        output_list = [layer.output for layer in base_model.layers if layer.name in layer_list]

    model = Model(inputs=base_model.input, outputs=output_list)

    return model

def compute_rep(input_tensor, model, preprocess_func=None):

    if preprocess_func is not None:
        input_tensor = preprocess_func(input_tensor.astype(np.float32))

    return model.predict(input_tensor.astype(np.float32))


def get_model_features(stim, model_type, preprocess):

    print "Computing model representation"
    model = get_model_rep(model_type)
    ns_model_features = compute_rep(stim, model, preprocess)
    ns_model_features = [f.reshape((f.shape[0], -1)) for f in ns_model_features]
    layer_names = [layer.name for layer in model.layers]

    return ns_model_features, layer_names

def compute_similarity_matrices(model_features):

    print "Computing model similarity matrices using Pearson R"
    sim_mats = []
    for i, f in enumerate(model_features):
        try:
            s = sim_pearson(f.reshape((f.shape[0],-1)))
        except Exception as e:
            print i, f.shape
            raise e
        sim_mats.append(s)
    # sim_mats = [sim_pearson(f.reshape((f.shape[0],-1))) for f in model_features]

    return sim_mats

def sim_pearson(X):
    # X is [dim, samples]
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1)) + 1e-7

    cor = np.dot(dX, dX.T)/(dX.shape[1]*sigma)
    cor = (cor.T/sigma).T

    return cor

def shuffle_sim_mat(sim_mat):

    n, _ = sim_mat.shape

    p = np.random.permutation(n)

    s = sim_mat[p]
    s = (s.T[p]).T

    return s

def compute_ssm(rep1, rep2, num_shuffles=None, num_folds=None):

    def comparison_function(r1, r2):
        r, _ = spearmanr(r1.flatten(), r2.flatten())
        return r

    rfolds = np.zeros(num_folds)
    rshuffles = np.zeros(num_shuffles)

    try:
        r = comparison_function(rep1, rep2)
    except ValueError as ve:
        print rep1.shape, rep2.shape
        raise ve

    if num_folds is not None:
        fold_size = rep1.shape[0]/num_folds + (1 if rep1.shape[0]%num_folds!=0 else 0)
        for n in xrange(num_folds):
            lower = fold_size*n
            upper = lower + fold_size
            temp_rep1 = rep1[lower:upper]
            temp_rep1 = (temp_rep1.T[lower:upper]).T

            temp_rep2 = rep2[lower:upper]
            temp_rep2 = (temp_rep2.T[lower:upper]).T

            rtemp = comparison_function(temp_rep1, temp_rep2)
            rfolds[n] = rtemp
    else:
        rfolds = None

    if num_shuffles is not None:
        for n in xrange(num_shuffles):
            shuffle_rep1 = shuffle_sim_mat(rep1)
            shuffle_rep2 = shuffle_sim_mat(rep2)
            rtemp = comparison_function(shuffle_rep1, shuffle_rep2)

            rshuffles[n] = rtemp
    else:
        rshuffles = None

    return r, rfolds, rshuffles

def compute_ssm_for_model(sim_mats, num_shuffles=None, num_folds=None):

    # compute SSM for model layers
    model_ssm = np.zeros((len(sim_mats), len(sim_mats)))
    if num_shuffles is not None:
        model_ssm_shuffles = np.zeros(model_ssm.shape + (num_shuffles,))
    else:
        model_ssm_shuffles = None
    if num_folds is not None:
        model_ssm_folds = np.zeros(model_ssm.shape+ (num_folds,))
    else:
        model_ssm_folds = None

    for i, mat_i in enumerate(sim_mats):
        for j, mat_j in enumerate(sim_mats):

            r, rfolds, rshuffles = compute_ssm(mat_i, mat_j, num_shuffles=num_shuffles, num_folds=num_folds)

            model_ssm[i,j] = r
            if num_folds is not None:
                model_ssm_folds[i,j] = rfolds
            if num_shuffles is not None:
                model_ssm_shuffles[i,j] = rshuffles

    return model_ssm, model_ssm_folds, model_ssm_shuffles


class ModelProperties (object):
    def __init__(self, model_type, preprocess, stim, compute_sim=True):
        self.model_type = model_type
        self.preprocess = preprocess
        self.stim = stim

        self.model_features, self.layer_names = get_model_features(self.stim,
                                                        model_type=self.model_type,
                                                        preprocess=self.preprocess)

        if compute_sim:
            self.sim_mats = compute_similarity_matrices(self.model_features)

            self.model_ssm, _, _ = compute_ssm_for_model(self.sim_mats)

            dist = 1.0 - self.model_ssm
            y = dist[np.triu_indices(dist.shape[0],k=1)]
            self.Z = linkage(y, 'ward')

            self.cluster_labels = fcluster(self.Z,5,criterion='maxclust')
        else:
            self.sim_mats = None
            self.model_ssm = None
            self.Z = None
            self.cluster_labels = None


class AlexNetProperties (object):
    def __init__(self, stim):

        base_model = AlexNet(weights_path='/home/michaelbu/Code/convnets-keras/alexnet_weights_new.h5', heatmap=False)

        self.model_type = 'AlexNet'
        self.preprocess = preprocess_input_16
        self.stim = stim

        # self.model_features, self.layer_names = get_model_features(self.stim,
        #                                                 model_type=self.model_type,
        #                                                 preprocess=self.preprocess)

        output_list = [layer.output for layer in base_model.layers]

        model = Model(inputs=base_model.input, outputs=output_list)

        ns_model_features = compute_rep(stim, model, self.preprocess)
        ns_model_features = [f.reshape((f.shape[0], -1)) for f in ns_model_features]

        self.model_features = ns_model_features

        self.layer_names = ['input_2',
                            'conv_1',
                            'max_pooling2d_3',
                            'convpool_1',
                            'zero_padding2d_5',
                            'lambda_8',
                            'lambda_9',
                            'conv_2_1',
                            'conv_2_2',
                            'conv_2',
                            'max_pooling2d_4',
                            'lambda_10',
                            'zero_padding2d_6',
                            'conv_3',
                            'zero_padding2d_7',
                            'lambda_11',
                            'lambda_12',
                            'conv_4_1',
                            'conv_4_2',
                            'conv_4',
                            'zero_padding2d_8',
                            'lambda_13',
                            'lambda_14',
                            'conv_5_1',
                            'conv_5_2',
                            'conv_5',
                            'convpool_5',
                            'flatten',
                            'dense_1',
                            'dropout_3',
                            'dense_2',
                            'dropout_4',
                            'dense_3',
                            'softmax']

        self.sim_mats = compute_similarity_matrices(self.model_features)

        self.model_ssm, _, _ = compute_ssm_for_model(self.sim_mats)

        dist = 1.0 - self.model_ssm
        y = dist[np.triu_indices(dist.shape[0],k=1)]
        self.Z = linkage(y, 'ward')

        self.cluster_labels = fcluster(self.Z,5,criterion='maxclust')


def cluster_model(input_size):

    ns_stim = get_stim(new_size=input_size)
    print "ns_stim.shape=", ns_stim.shape

    # model_props = ModelProperties(ResNet50, preprocess_input_resnet, ns_stim)
    # model_name = 'resnet50'
    # model_props = ModelProperties(InceptionV3, preprocess_input_inception, ns_stim)
    # model_name = 'inceptionv3'
    # model_props = ModelProperties(VGG19, preprocess_input_19, ns_stim)
    # model_name = 'vgg19'
    # model_props = ModelProperties(VGG16, preprocess_input_16, ns_stim)
    # model_name = 'vgg16'
    # model_props = LeMouseProperties()
    # model_name = 'lemouse'
    model_props = AlexNetProperties(ns_stim)
    model_name = 'alexnet'
    print model_props.cluster_labels

    clusters = h5py.File(model_name+'_clusters_'+str(input_size[0])+'.h5', 'w')
    clusters['clusters'] = model_props.cluster_labels
    clusters.close()

    print model_props.cluster_labels

    # dendrogram(model_props.Z)
    # plt.figure()
    # plt.imshow(model_props.model_ssm)

    # plt.show()


if __name__=='__main__':

    # cluster_model((50,64))
    # cluster_model((100,128))
    # cluster_model((200,256))
    # cluster_model((400,512))

    cluster_model((227,227))
