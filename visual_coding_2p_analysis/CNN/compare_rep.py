import numpy as np
import os, sys
import pickle
import h5py


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

# from keras.models import Model

from ModelProperties import ModelProperties, compute_ssm, AlexNetProperties
from bob_response import compute_response_dict_from_bob, response_dict_by_layer_from_depth
from bob_stim import get_stim

RESPONSE_CACHE = 'bob_ns_response_dict.pkl'

# NEW_SIZE = (400,512)
# NEW_SIZE= (200,256)
# NEW_SIZE= (100,128)
# NEW_SIZE = (50,64)
NEW_SIZE = (227,227)
IMAGENET = True

NUM_FOLDS = 4
NUM_SHUFFLES = 100

PREPROCESS_INPUT = preprocess_input_16
MODEL_TYPE = VGG16
MODEL_NAME = 'vgg16'

# PREPROCESS_INPUT = preprocess_input_19
# MODEL_TYPE = VGG19
# MODEL_NAME = 'vgg19'

# PREPROCESS_INPUT = preprocess_input_inception
# MODEL_TYPE = InceptionV3
# MODEL_NAME = 'inceptionv3'

# PREPROCESS_INPUT = preprocess_input_resnet50
# MODEL_TYPE = ResNet50
# MODEL_NAME = 'resnet50'

# PREPROCESS_INPUT = preprocess_input_16
# MODEL_TYPE = 'AlexNet'
# MODEL_NAME = 'alexnet'

EXCITATORY = ['Emx1', 'Slc17a7', 'Cux2', 'Rorb', 'Scnn1a', 'Nr5a1', 'Rbp4', 'Fezf2', 'Tlx3', 'Ntsr1']
INHIBITORY = ['Sst', 'Vip', 'Pvalb']




# def get_base_model(model_class):

#     if IMAGENET:
#         weights='imagenet'
#     else:
#         weights=None
#     base_model = model_class(weights=weights, include_top=False)

#     return base_model

# def get_model_rep(model_class, layer_list=None):

#     base_model = get_base_model(model_class)
#     # print base_model.layers

#     if layer_list is None:
#         output_list = [layer.output for layer in base_model.layers]
#     else:
#         output_list = [layer.output for layer in base_model.layers if layer.name in layer_list]

#     model = Model(inputs=base_model.input, outputs=output_list)

#     return model


# def compute_rep(input_tensor, model, preprocess_func=None):

#     if preprocess_func is not None:
#         input_tensor = preprocess_func(input_tensor.astype(np.float32))

#     return model.predict(input_tensor.astype(np.float32))

def sim_pearson(X):
    # X is [dim, samples]
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1))

    cor = np.dot(dX, dX.T)/(dX.shape[1]*(sigma+1e-7))
    cor = (cor.T/(sigma+1e-7)).T

    return cor



# def get_model_features(stim, model_type=MODEL_TYPE, preprocess=PREPROCESS_INPUT):

#     print "Computing model representation"
#     model = get_model_rep(model_type)
#     ns_model_features = compute_rep(stim, model, preprocess)
#     ns_model_features = [f.reshape((f.shape[0], -1)) for f in ns_model_features]
#     layer_names = [layer.name for layer in model.layers]

#     return ns_model_features, layer_names

# def compute_similarity_matrices(model_features):

#     print "Computing model similarity matrices using Pearson R"
#     sim_mats = [sim_pearson(f.reshape((f.shape[0],-1))) for f in model_features]

#     return sim_mats








# def compute_ssm(rep1, rep2, num_shuffles, num_folds):

#     def comparison_function(r1, r2):
#         r, _ = spearmanr(r1.flatten(), r2.flatten())
#         return r

#     rfolds = np.zeros(num_folds)
#     rshuffles = np.zeros(num_shuffles)

#     r = comparison_function(rep1, rep2)

#     fold_size = rep1.shape[0]/num_folds + (1 if rep1.shape[0]%num_folds!=0 else 0)
#     for n in xrange(num_folds):
#         lower = fold_size*n
#         upper = lower + fold_size
#         temp_rep1 = rep1[lower:upper]
#         temp_rep1 = (temp_rep1.T[lower:upper]).T

#         temp_rep2 = rep2[lower:upper]
#         temp_rep2 = (temp_rep2.T[lower:upper]).T

#         rtemp = comparison_function(temp_rep1, temp_rep2)
#         rfolds[n] = rtemp

#     for n in xrange(num_shuffles):
#         shuffle_rep1 = shuffle_sim_mat(rep1)
#         shuffle_rep2 = shuffle_sim_mat(rep2)
#         rtemp = comparison_function(shuffle_rep1, shuffle_rep2)

#         rshuffles[n] = rtemp

#     return r, rfolds, rshuffles


# def compute_ssm_for_model(sim_mats, num_shuffles, num_folds):

#     # compute SSM for model layers
#     model_ssm = np.zeros((len(sim_mats), len(sim_mats)))
#     model_ssm_shuffles = np.zeros(model_ssm.shape + (num_shuffles,))
#     model_ssm_folds = np.zeros(model_ssm.shape+ (num_folds,))
#     for i, mat_i in enumerate(sim_mats):
#         for j, mat_j in enumerate(sim_mats):

#             r, rfolds, rshuffles = compute_ssm(mat_i, mat_j, num_shuffles=num_shuffles, num_folds=num_folds)

#             model_ssm[i,j] = r
#             model_ssm_folds[i,j] = rfolds
#             model_ssm_shuffles[i,j] = rshuffles

#     return model_ssm, model_ssm_folds, model_ssm_shuffles

def compare_reps(input_size):

    ns_stim = get_stim(new_size=input_size)
    # ns_model_features, layer_names = get_model_features(ns_stim)
    # sim_mats = compute_similarity_matrices(ns_model_features)

    if MODEL_TYPE=='AlexNet':
        model_props = AlexNetProperties(ns_stim)
    else:
        model_props = ModelProperties(MODEL_TYPE, PREPROCESS_INPUT, ns_stim)
    layer_names = model_props.layer_names
    sim_mats = model_props.sim_mats

    # SVD is used for SVCCA; not computing this because sample size is to small
    # model_svd = [SVD(f) for f in ns_model_features]

    save_file = h5py.File(MODEL_NAME+'_ssm_cca_'+str(input_size[0])+'.h5','w')
    save_file['num_shuffles'] = NUM_SHUFFLES
    save_file['num_folds'] = NUM_FOLDS

    # model_ssm, model_ssm_folds, model_ssm_shuffles = compute_ssm_for_model(sim_mats,
    #                                                     num_shuffles=NUM_SHUFFLES,
    #                                                     num_folds=NUM_FOLDS)

    model_ssm = model_props.model_ssm

    save_file['model_ssm/r'] = model_ssm
    # save_file['model_ssm/rfolds'] = model_ssm_folds
    # save_file['model_ssm/rshuffles'] = model_ssm_shuffles

    if os.path.exists(RESPONSE_CACHE):
        print "Opening cached response_dict"
        with open(RESPONSE_CACHE, 'r') as f:
            response_dict = pickle.load(f)
    else:
        print "Computing response_dict, saving to cache"
        response_dict = compute_response_dict_from_bob()
        with open(RESPONSE_CACHE, 'w') as f:
            pickle.dump(response_dict, f)

    response_dict = response_dict_by_layer_from_depth(response_dict)

    print "Computing SVCCA and SSM results and saving to file"
    for area in response_dict.keys():
        resp_exc = {'layer23':[], 'layer4':[], 'layer5':[], 'layer6':[]}
        resp_inh = {'layer23':[], 'layer4':[], 'layer5':[], 'layer6':[]}
        for cre in response_dict[area].keys():
            for layer in response_dict[area][cre].keys():
                print area, cre, layer
                print response_dict[area][cre][layer].shape

                response = response_dict[area][cre][layer]

                if cre in EXCITATORY:
                    resp_exc[layer] += [response]
                if cre in INHIBITORY:
                    resp_inh[layer] += [response]

                # svd_r = SVD(response)

                sim_resp = sim_pearson(response)
                save_file[area+'/'+cre+'/'+layer+'/sim_mat'] = sim_resp
                # do cca on this rep with model rep
                for i, model_layer in enumerate(layer_names):
                    # svcca = SVCCA(response, ns_model_features[i])
                    # svcca = SVCCA(svd_r, model_svd[i])
                    # rho = svcca.rho
                    # save_file[area+'/'+cre+'/'+layer+'/'+str(i)+'_'+model_layer+'/rho'] = rho

                    sm = sim_mats[i]

                    r, rfolds, rshuffles = compute_ssm(sm, sim_resp, num_shuffles=NUM_SHUFFLES, num_folds=NUM_FOLDS)
                    # sp_r, _ = spearmanr(sm.flatten(), sim_resp.flatten())

                    save_file[area+'/'+cre+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/r'] = r
                    save_file[area+'/'+cre+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rfolds'] = rfolds
                    save_file[area+'/'+cre+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rshuffles'] = rshuffles

        for layer in resp_exc:

            if len(resp_exc[layer])>0:
                resp_e = np.hstack(resp_exc[layer])
                # svd_e = SVD(resp_e)
                sim_resp = sim_pearson(resp_e)
                for i, model_layer in enumerate(layer_names):
                    # svcca = SVCCA(response, ns_model_features[i])
                    # svcca = SVCCA(svd_e, model_svd[i])
                    # rho = svcca.rho
                    # save_file[area+'/'+'excitatory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/rho'] = rho

                    sm = sim_mats[i]
                    # sp_r, _ = spearmanr(sm.flatten(), sim_resp.flatten())
                    r, rfolds, rshuffles = compute_ssm(sm, sim_resp, num_shuffles=NUM_SHUFFLES, num_folds=NUM_FOLDS)

                    save_file[area+'/'+'excitatory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/r'] = r
                    save_file[area+'/'+'excitatory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rfolds'] = rfolds
                    save_file[area+'/'+'excitatory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rshuffles'] = rshuffles

            if len(resp_inh[layer])>0:
                resp_i = np.hstack(resp_inh[layer])
                # svd_i = SVD(resp_i)
                sim_resp = sim_pearson(resp_i)
                for i, model_layer in enumerate(layer_names):
                    # svcca = SVCCA(response, ns_model_features[i])
                    # svcca = SVCCA(svd_i, model_svd[i])
                    # rho = svcca.rho
                    # save_file[area+'/'+'inhibitory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/rho'] = rho

                    sm = sim_mats[i]
                    # sp_r, _ = spearmanr(sm.flatten(), sim_resp.flatten())
                    r, rfolds, rshuffles = compute_ssm(sm, sim_resp, num_shuffles=NUM_SHUFFLES, num_folds=NUM_FOLDS)

                    save_file[area+'/'+'inhibitory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/r'] = r
                    save_file[area+'/'+'inhibitory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rfolds'] = rfolds
                    save_file[area+'/'+'inhibitory'+'/'+layer+'/'+str(i)+'_'+model_layer+'/ssm/rshuffles'] = rshuffles

        # print response.shape
        # print response
        #print stim_table



if __name__ == '__main__':

    # NEW_SIZE = (400,512)
    # NEW_SIZE= (200,256)
    # NEW_SIZE= (100,128)
#    NEW_SIZE = (50,64)

#    compare_reps(input_size=NEW_SIZE)

    compare_reps(input_size=(50,64))
    # compare_reps(input_size=(100,128))
    # compare_reps(input_size=(200,256))
    # compare_reps(input_size=(400,512))


    # compare_reps(input_size=(227,227))
