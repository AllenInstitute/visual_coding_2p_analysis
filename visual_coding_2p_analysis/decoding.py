### code for classifying the stimulus

import os, sys, pickle, h5py
import numpy as np
from pandas import HDFStore
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import sklearn.mixture as mixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from scipy.stats import ttest_rel, ttest_1samp, spearmanr

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from visual_coding_2p_analysis.l0_analysis import L0_analysis
from visual_coding_2p_analysis import core
from visual_coding_2p_analysis.correlations import label_run_stationary_dpgmm

import gaussClassifier as gc



manifest_file = core.get_manifest_path()
boc = BrainObservatoryCache(manifest_file=manifest_file)


def fitGauss(r,lam=0.1):
    '''r is a [T,N] matrix where T is the number of trials and N is the dimension of the response'''

    m = np.mean(r,axis=0)

    T,N = r.shape
    delta = r - m
    C = np.dot(delta.T,delta)/T + np.eye(N)*lam

    return m,C


def fitGaussIndependent(r,lam = 0.1):
    '''r is a [T,N] matrix where T is the number of trials and N is the dimension of the response
        lam is the coefficient of an L2 regularizer'''

    m = np.mean(r,axis=0)

    T,N = r.shape

    delta = r - m
    C = np.sum(delta*delta,axis=0)/T + lam
    C = np.diag(C)

    return m,C


def gaussLikelihood(r,m,C):
    '''log likelihood for gaussian with mean m and covariance C with data r (a [T,N] matrix where T is trials and N is dimension of response'''

    T,N = r.shape
    delta = r - m  # will be [T,N]


    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.linalg.LinAlgError as error:
        #print error
        #print C
        #print np.diagonal(C)
        #print r
        raise(error)


    A = -0.5*np.trace(np.dot(delta,np.dot(Cinv,delta.T)))
    B = -0.5*T*np.log(np.linalg.det(C))

    return A+B  #ignoring constant -0.5*TN*log(2.0*np.pi)


def gaussLikelihoodForCategories(responses,classSamples,fitFunc=fitGauss,lam=0.1):

    classLabels = np.unique(classSamples)

    classParams = {}
    for c in classLabels:
        mask = (classSamples == c)
        #print mask
        r = responses[mask]
        #print r

        m, C = fitFunc(r,lam=lam)

        classParams[c] = (m,C)

    return classParams


def computeClassProb(classSamples):
    '''classLabels is a vector of class labels of length T (the number of samples)'''

    classProb = {}

    for c in classSamples:
        classProb[c] = classProb.get(c,0) + 1

    N = float(len(classSamples))

    for c in classProb:
        classProb[c] = classProb[c]/N

    return classProb


class GDA(object):
    def __init__(self,responseTrain,classTrain,num_factors,lam=0.1,OnevAll=False, shrinkage='ledoit_wolf'):

        self.lam = lam
        self.OnevAll = OnevAll
        self.shrinkage = shrinkage
        self.classes = np.unique(classTrain)
        self.numClasses = len(np.unique(classTrain))
        self.defineClassIDs()

        self.classProb = self.computeClassProb(classTrain)
        self.computeClassParams(responseTrain,classTrain,num_factors)

        #self.classParams = gaussLikelihoodForCategories(responseTrain,classTrain,fitFunc=fitGaussIndependent,lam=lam)

        self.responseTrain = responseTrain
        self.classTrain = classTrain
        self.defineClassIDs()

        #try using shared covariance, override previous computation
        #C = np.dot(responseTrain.T,responseTrain)/responseTrain.shape[0]
        #C = np.diag(np.diagonal(C))
        #for c in self.classParams:
        #    m, Cov = self.classParams[c]
        #    self.classParams[c] = (m,C)

    def computeClassParams(self,responseTrain,classTrain,num_factors):

        T, N = responseTrain.shape

        self.class_means = np.zeros([self.numClasses,N])
        self.class_inv_covs = np.zeros([self.numClasses,N,N])
        self.class_inv_cov_dets = np.zeros(self.numClasses)
        #estimate class means, self.class_means.shape = (numClasses, N)

        if num_factors == None or num_factors == 0:

            class_avg_cov = np.zeros((N, N))

            for i,c in enumerate(self.classes):

                rClass = responseTrain[classTrain==c]
                self.class_means[self.classIDs[c]] = np.mean(rClass, axis=0)

                if self.shrinkage == 'ledoit-wolf':
                    class_cov, _ = ledoit_wolf(rClass)
                    self.class_inv_covs[i] = np.linalg.pinv(class_cov, rcond=1e-9)
                    self.class_inv_cov_dets[i] = np.linalg.det(self.class_inv_covs[i])

                elif self.shrinkage == 'diagonal':
                    delta_class_response = rClass - self.class_means[i]
                    class_cov = np.dot(delta_class_response.T, delta_class_response) / float(T)
                    class_cov = (1-self.lam)*class_cov + self.lam*np.eye(N)
                    # self.class_inv_covs[i] = np.linalg.inv(class_cov)
                    self.class_inv_covs[i] = np.linalg.pinv(class_cov, rcond=1e-9)
                    self.class_inv_cov_dets[i] = np.linalg.det(self.class_inv_covs[i])


                elif self.shrinkage == 'classMean':
                    delta_class_response = rClass - self.class_means[i]
                    class_cov = np.dot(delta_class_response.T, delta_class_response) / float(T)
                    class_avg_cov += class_cov

                else:
                    delta_class_response = rClass - self.class_means[i]
                    class_cov = np.dot(delta_class_response.T, delta_class_response) / float(T)
                    # self.class_inv_covs[i] = np.linalg.inv(class_cov)
                    self.class_inv_covs[i] = np.linalg.pinv(class_cov, rcond=1e-9)
                    self.class_inv_cov_dets[i] = np.linalg.det(self.class_inv_covs[i])

            if self.shrinkage == 'classMean':
                class_avg_cov /= float(len(self.classes))
                for i,c in enumerate(self.classes):

                    rClass = responseTrain[classTrain == c]
                    delta_class_response = rClass - self.class_means[i]
                    class_cov = np.dot(delta_class_response.T, delta_class_response) / float(T)
                    class_cov = (1-self.lam)*class_cov + self.lam*class_avg_cov

                    # self.class_inv_covs[i] = np.linalg.inv(class_cov)
                    self.class_inv_covs[i] = np.linalg.pinv(class_cov, rcond=1e-9)
                    self.class_inv_cov_dets[i] = np.linalg.det(self.class_inv_covs[i])


        else:
            fa = decompose.FactorAnalysis(n_components=num_factors)

            for i, c in enumerate(self.classes):

                rClass = responseTrain[classTrain==c]

                fa.fit(rClass)
                factor = fa.transform(rClass)

                self.class_means[self.classIDs[c]] = np.mean(rClass,axis=0)

                factor_cov = factor.T.dot(factor)
                class_cov = fa.components_.T.dot(factor_cov).dot(fa.components_) + np.diag(fa.noise_variance_)

                # self.class_inv_covs[i] = np.linalg.inv(class_cov)
                self.class_inv_covs[i] = np.linalg.pinv(class_cov, rcond=1e-9)
                self.class_inv_cov_dets[i] = np.linalg.det(self.class_inv_covs[i])


    def fit(self,responseTest):
        '''compute probability of being in class c, for each c'''


        T, N = responseTest.shape

        log_prob = np.zeros([T,self.numClasses])
        for i,c in enumerate(self.classes):
            delta_class_response = responseTest - self.class_means[i]  #(T,N)
            log_prob.T[i] = -0.5*np.sum((np.dot(delta_class_response, self.class_inv_covs[i])*delta_class_response),axis=1)
            log_prob.T[i] = log_prob.T[i] + 0.5*np.log(self.class_inv_cov_dets[i])

        log_prob = log_prob + np.log(self.classProb)

        self.log_probs = log_prob

        maxLogs = np.max(self.log_probs,axis=1)  #shape = T
        delta = (self.log_probs.T  - maxLogs)  #shape = (numClasses,T)
        logNorm = np.log(np.sum(np.exp(delta),axis=0)) + maxLogs  #shape = (T)

        self.posterior_probs = np.exp(self.log_probs.T - logNorm).T

        return self.posterior_probs

    def predict(self,responseTest):

        #going to assume fit has been calculated
        if self.OnevAll:
            for i in range(self.numClasses):
                ind = np.where(np.arange(self.numClasses) != i)[0]
                self.posterior_probs[:, i] /= np.sum(self.posterior_probs[:, ind], axis=1)
            prediction_args = np.argmax(self.posterior_probs, axis=1)
            return np.array([self.classes[j] for j in prediction_args])

        else:
            prediction_args = np.argmax(self.posterior_probs,axis=1)
            return np.array([self.classes[i] for i in prediction_args])



    def computeClassProb(self,classSamples):
        '''classSamples is a vector of class labels of length T (the number of samples)'''

        self.classProb = np.zeros(len(self.classes))
        classProbDict = {}

        for c in classSamples:
            classProbDict[c] = classProbDict.get(c,0) + 1

        N = float(len(classSamples))

        for c in classProbDict:
            self.classProb[self.classIDs[c]] = classProbDict[c]/float(N)

        return self.classProb

    def defineClassIDs(self):

        classIDs = {}
        for i,c in enumerate(self.classes):
            classIDs[c] = i

        self.classIDs = classIDs


class NaiveBayes (object):
    def __init__(self,responseTrain,classTrain,lam=0.1,OnevAll=False):

        self.lam = lam
        self.OnevAll = OnevAll
        self.classes = np.unique(classTrain)
        self.numClasses = len(np.unique(classTrain))
        self.defineClassIDs()

        self.classProb = self.computeClassProb(classTrain)
        self.computeClassParams(responseTrain,classTrain)

        #self.classParams = gaussLikelihoodForCategories(responseTrain,classTrain,fitFunc=fitGaussIndependent,lam=lam)

        self.responseTrain = responseTrain
        self.classTrain = classTrain



        #try using shared covariance, override previous computation
        #C = np.dot(responseTrain.T,responseTrain)/responseTrain.shape[0]
        #C = np.diag(np.diagonal(C))
        #for c in self.classParams:
        #    m, Cov = self.classParams[c]
        #    self.classParams[c] = (m,C)

    def computeClassParams(self,responseTrain,classTrain):

        T,N = responseTrain.shape

        self.class_means = np.zeros([self.numClasses,N])
        self.class_inv_covs = np.zeros([self.numClasses,N])
        #estimate class means, self.class_means.shape = (numClasses, N)
        for i,c in enumerate(self.classes):
            self.class_means[self.classIDs[c]] = np.mean(responseTrain[classTrain==c],axis=0)

            delta_class_response = responseTrain[classTrain==c] - self.class_means[i]
            class_cov = np.sum(delta_class_response**2,axis=0)/float(T) + self.lam
            self.class_inv_covs[i] = 1.0/class_cov

    def fit(self,responseTest):
        '''compute probability of being in class c, for each c'''


        T, N = responseTest.shape

        log_prob = np.zeros([T,self.numClasses])
        for i,c in enumerate(self.classes):
            delta_class_response = responseTest - self.class_means[i]
            log_prob.T[i] = -0.5*np.dot(delta_class_response**2, self.class_inv_covs[i])

        log_prob = log_prob +  0.5*np.sum(np.log(self.class_inv_covs),axis=1)
        log_prob = log_prob + np.log(self.classProb)

        self.log_probs = log_prob

        maxLogs = np.max(self.log_probs,axis=1)  #shape = T
        delta = (self.log_probs.T  - maxLogs)  #shape = (numClasses,T)
        logNorm = np.log(np.sum(np.exp(delta),axis=0)) + maxLogs  #shape = (T)

        self.posterior_probs = np.exp(self.log_probs.T - logNorm).T

        return self.posterior_probs

    def predict(self,responseTest):

        #going to assume fit has been calculated
        if self.OnevAll:
            for i in range(self.numClasses):
                ind = np.where(np.arange(self.numClasses) != i)[0]
                self.posterior_probs[:, i] /= np.sum(self.posterior_probs[:, ind], axis=1)
            prediction_args = np.argmax(self.posterior_probs, axis=1)
            return np.array([self.classes[j] for j in prediction_args])

        else:
            prediction_args = np.argmax(self.posterior_probs,axis=1)
            return np.array([self.classes[i] for i in prediction_args])


    def computeClassProb(self,classSamples):
        '''classSamples is a vector of class labels of length T (the number of samples)'''

        self.classProb = np.zeros(len(self.classes))
        classProbDict = {}

        for c in classSamples:
            classProbDict[c] = classProbDict.get(c,0) + 1

        N = float(len(classSamples))

        for c in classProbDict:
            self.classProb[self.classIDs[c]] = classProbDict[c]/float(N)

        return self.classProb

    def defineClassIDs(self):

        classIDs = {}
        for i,c in enumerate(self.classes):
            classIDs[c] = i

        self.classIDs = classIDs


class LDA (object):
    def __init__(self,responseTrain,classTrain,num_factors,lam=0.1,OnevAll=False, shrinkage='diagonal'):
        self.classes = np.unique(classTrain)
        self.numClasses = len(self.classes)

        self.lam = lam
        self.shrinkage=shrinkage
        self.OnevAll = OnevAll

        self.defineClassIDs()

        self.computeClassProb(classTrain)
        self.computeClassParams(responseTrain,classTrain,num_factors)


    def defineClassIDs(self):
        '''these classIDs are internal, not like the other two classifiers here!'''
        classIDs = {}
        for i,c in enumerate(self.classes):
            classIDs[c] = i

        self.classIDs = classIDs


    def computeClassParams(self,responseTrain,classTrain,num_factors):

        T,N = responseTrain.shape
        self.class_means = np.zeros([self.numClasses, N])

        #estimate class means
        # IS THERE A BUG HERE!!!

        for i, c in enumerate(self.classes):
            rClass = responseTrain[classTrain == c]
            self.class_means[self.classIDs[c]] = np.mean(rClass, axis=0)

        if num_factors != None and num_factors != 0:

            fa = decompose.FactorAnalysis(n_components=num_factors, svd_method='lapack')
            fa.fit(responseTrain)
            cov_fa = fa.get_covariance()
            self.cov = cov_fa

        else:

            if self.shrinkage == 'ledoit-wolf':
                self.cov, _ = ledoit_wolf(responseTrain)

            elif self.shrinkage == 'diagonal':
                cov_mle = np.zeros([N, N])
                for i, c in enumerate(self.classes):
                    rClass = responseTrain[classTrain == c]
                    rClass = rClass - self.class_means[self.classIDs[c]]   #shouldn't we subtract the mean as well?
                    cov_mle += np.dot(rClass.T,rClass)/(T-self.numClasses)

                # self.cov = self.lam*np.diag(np.diag(cov_mle)) + (1.0 - self.lam)*cov_mle
                self.cov = cov_mle + self.lam * np.eye(cov_mle.shape[0])


            else:
                cov_mle = np.zeros([N, N])
                for i, c in enumerate(self.classes):
                    rClass = rClass - self.class_means[self.classIDs[c]]   #shouldn't we subtract the mean as well?
                    cov_mle += np.dot(rClass.T,rClass)/(T-self.numClasses)

                self.cov = cov_mle


        self.cov_inv = np.linalg.inv(self.cov)
        #self.cov_inv = np.eye(N)

        self.gamma = -0.5*np.diagonal(np.dot(self.class_means, np.dot(self.cov_inv, self.class_means.T))) + np.log(self.classProb)  # shape = (numClasses)

        self.beta = np.dot(self.cov_inv, self.class_means.T).T  #shape = [numClasses, N]


    def bootstrapClassParams(self, responseTrain, classTrain, num_factors, boots=100):
        # NOT FINISHED
        T, N = responseTrain.shape
        self.bootstrapMeans = np.zeros([boots, self.numClasses, N])
        self.bootstrapCovs = np.zeros([boots, self.numClasses, N, N])

        for b in range(boots):

            trials = np.random.choice(T, size=T, replace=True)
            responseTemp = responseTrain[trials]
            classTemp = classTrain[trials]

            for i, c in enumerate(self.classes):
                rClass = responseTemp[classTemp == c]
                self.bootstrapMeans[b, self.ClassIDs[c]] = np.mean(rClass, axis=0)

            if num_factors != None and num_factors != 0:

                fa = decompose.FactorAnalysis(n_components=num_factors)
                fa.fit(responseTemp)
                cov_fa = fa.get_covariance()


            else: # fix this for bootstrapping

                if self.shrinkage == 'ledoit-wolf':
                    self.cov, _ = ledoit_wolf(responseTrain)

                elif self.shrinkage == 'diagonal':
                    cov_mle = np.zeros([N, N])
                    for i, c in enumerate(self.classes):
                        rClass = rClass - self.class_means[
                            self.classIDs[c]]  # shouldn't we subtract the mean as well?
                        cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                    # self.cov = self.lam * np.diag(np.diag(cov_mle)) + (1.0 - self.lam) * cov_mle
                    self.cov += self.lam*np.eye(N)

                else:
                    cov_mle = np.zeros([N, N])
                    for i, c in enumerate(self.classes):
                        rClass = rClass - self.class_means[
                            self.classIDs[c]]  # shouldn't we subtract the mean as well?
                        cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                    self.cov = cov_mle

        self.class_means = np.zeros([self.numClasses, N])

        # estimate class means
        # IS THERE A BUG HERE!!!

        for i, c in enumerate(self.classes):
            rClass = responseTrain[classTrain == c]
            self.class_means[self.classIDs[c]] = np.mean(rClass, axis=0)

        if num_factors != None and num_factors != 0:

            fa = decompose.FactorAnalysis(n_components=num_factors)
            fa.fit(responseTrain)
            cov_fa = fa.get_covariance()
            self.cov = cov_fa

        else:

            if self.shrinkage == 'ledoit-wolf':
                self.cov, _ = ledoit_wolf(responseTrain)

            elif self.shrinkage == 'diagonal':
                cov_mle = np.zeros([N, N])
                for i, c in enumerate(self.classes):
                    rClass = rClass - self.class_means[self.classIDs[c]]  # shouldn't we subtract the mean as well?
                    cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                self.cov = self.lam * np.diag(np.diag(cov_mle)) + (1.0 - self.lam) * cov_mle

            else:
                cov_mle = np.zeros([N, N])
                for i, c in enumerate(self.classes):
                    rClass = rClass - self.class_means[self.classIDs[c]]  # shouldn't we subtract the mean as well?
                    cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                self.cov = cov_mle

        self.cov_inv = np.linalg.inv(self.cov)
        # self.cov_inv = np.eye(N)

        self.gamma = -0.5 * np.diagonal(np.dot(self.class_means, np.dot(self.cov_inv, self.class_means.T))) + np.log(
            self.classProb)  # shape = (numClasses)

        self.beta = np.dot(self.cov_inv, self.class_means.T).T  # shape = [numClasses, N]

    def computeClassProb(self,classSamples):
        '''classLabels is a vector of class labels of length T (the number of samples)'''

        self.classProb = np.zeros(len(self.classes))
        classProbDict = {}

        for c in classSamples:
            classProbDict[c] = classProbDict.get(c, 0) + 1

        N = float(len(classSamples))

        for c in classProbDict:
            self.classProb[self.classIDs[c]] = classProbDict[c]/float(N)

        return self.classProb

    def fit(self,responseTest):

        #responseTest should be shape = (T,N)

        self.log_probs = np.dot(responseTest, self.beta.T) + self.gamma  #shape = (T,numClasses)

        maxLogs = np.max(self.log_probs,axis=1)  #shape = T
        delta = (self.log_probs.T  - maxLogs)  #shape = (numClasses,T)
        logNorm = np.log(np.sum(np.exp(delta),axis=0)) + maxLogs  #shape = (T)

        self.posterior_probs = np.exp(self.log_probs.T - logNorm).T

        return self.posterior_probs

    def predict(self,responseTest):

        #going to assume fit has been calculated
        if self.OnevAll:
            for i in range(self.numClasses):
                ind = np.where(np.arange(self.numClasses) != i)[0]
                self.posterior_probs[:, i] /= np.sum(self.posterior_probs[:, ind], axis=1)
            prediction_args = np.argmax(self.posterior_probs, axis=1)
            return np.array([self.classes[j] for j in prediction_args])

        else:
            prediction_args = np.argmax(self.posterior_probs,axis=1)
            return np.array([self.classes[i] for i in prediction_args])


class bootstrapLDA (object):
    def __init__(self,responseTrain, classTrain, num_factors, lam=0.1, OnevAll=False, numBoots=100, shrinkage='diagonal'):
        self.classes = np.unique(classTrain)
        self.numClasses = len(self.classes)
        self.numBoots = numBoots
        self.lam = lam
        self.shrinkage=shrinkage
        self.OnevAll = OnevAll

        self.defineClassIDs()

        self.computeClassProb(classTrain)
        self.computeClassParams(responseTrain,classTrain,num_factors)


    def defineClassIDs(self):
        '''these classIDs are internal, not like the other two classifiers here!'''
        classIDs = {}
        for i,c in enumerate(self.classes):
            classIDs[c] = i

        self.classIDs = classIDs


    def draw_BootStrapSample(self, responseTrain, classTrain):

        '''
        draw bootstrap sample with same number of trials per class, putting classes in same location as original sample
        :param response:
        :param classes:
        :return:
        '''

        T, N = responseTrain.shape
        classes = np.unique(classTrain)
        rBoot = np.zeros((T, N))

        for i, c in enumerate(classes):

            ind = (classTrain == c)
            Tc = sum(ind)
            r = responseTrain[ind]

            boot = np.random.choice(range(Tc), size=Tc, replace=True)
            rBoot[ind] = r[boot]  # same class labels as original responseTrain

        return rBoot


    def computeClassParams(self,responseTrain,classTrain,num_factors):

        T,N = responseTrain.shape
        self.class_means = np.zeros([self.numClasses, N])
        self.class_means_boots = np.zeros([self.numClasses, N, self.numBoots])

        self.cov = np.zeros((N, N))
        self.cov_boots = np.zeros((N, N, self.numBoots))

        for n in range(self.numBoots):

            responseBoot = self.draw_BootStrapSample(responseTrain, classTrain)

            for i, c in enumerate(self.classes):
                rClass = responseBoot[classTrain == c]
                self.class_means_boots[i, :, n] = np.mean(rClass, axis=0)

            if num_factors != None and num_factors != 0:

                fa = decompose.FactorAnalysis(n_components=num_factors, svd_method='lapack')
                fa.fit(responseBoot)
                cov_fa = fa.get_covariance()
                self.cov_boots[:, :, n] = cov_fa

            else:

                if self.shrinkage == 'ledoit-wolf':
                    self.cov_boots[:, :, n], _ = ledoit_wolf(responseBoot)

                elif self.shrinkage == 'diagonal':
                    cov_mle = np.zeros([N, N])
                    for i, c in enumerate(self.classes):
                        rClass = responseBoot[classTrain == c]
                        rClass = rClass - self.class_means_boots[self.classIDs[c], :, n]
                        cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                    self.cov_boots[:, :, n] = cov_mle

                else:
                    cov_mle = np.zeros([N, N])
                    for i, c in enumerate(self.classes):
                        rClass = responseBoot[classTrain == c]
                        rClass = rClass - self.class_means[self.classIDs[c]]  # shouldn't we subtract the mean as well?
                        cov_mle += np.dot(rClass.T, rClass) / (T - self.numClasses)

                    self.cov_boots[:, :, n] = cov_mle

        self.class_means = np.mean(self.class_means_boots, axis=2)
        self.cov = np.mean(self.cov_boots, axis=2)

        if self.shrinkage == 'diagonal':
            self.cov += self.lam * np.eye(N)

        self.cov_inv = np.linalg.inv(self.cov)
        #self.cov_inv = np.eye(N)

        self.gamma = -0.5*np.diagonal(np.dot(self.class_means, np.dot(self.cov_inv, self.class_means.T))) + np.log(self.classProb)  # shape = (numClasses)
        self.beta = np.dot(self.cov_inv, self.class_means.T).T  #shape = [numClasses, N]


    def computeClassProb(self,classSamples):
        '''classLabels is a vector of class labels of length T (the number of samples)'''

        self.classProb = np.zeros(len(self.classes))
        classProbDict = {}

        for c in classSamples:
            classProbDict[c] = classProbDict.get(c, 0) + 1

        N = float(len(classSamples))

        for c in classProbDict:
            self.classProb[self.classIDs[c]] = classProbDict[c]/float(N)

        return self.classProb

    def fit(self,responseTest):

        #responseTest should be shape = (T,N)

        self.log_probs = np.dot(responseTest, self.beta.T) + self.gamma  #shape = (T,numClasses)

        maxLogs = np.max(self.log_probs,axis=1)  #shape = T
        delta = (self.log_probs.T  - maxLogs)  #shape = (numClasses,T)
        logNorm = np.log(np.sum(np.exp(delta),axis=0)) + maxLogs  #shape = (T)

        self.posterior_probs = np.exp(self.log_probs.T - logNorm).T

        return self.posterior_probs

    def predict(self,responseTest):

        #going to assume fit has been calculated
        if self.OnevAll:
            for i in range(self.numClasses):
                ind = np.where(np.arange(self.numClasses) != i)[0]
                self.posterior_probs[:, i] /= np.sum(self.posterior_probs[:, ind], axis=1)
            prediction_args = np.argmax(self.posterior_probs, axis=1)
            return np.array([self.classes[j] for j in prediction_args])

        else:
            prediction_args = np.argmax(self.posterior_probs,axis=1)
            return np.array([self.classes[i] for i in prediction_args])


class KNN (object):
    def __init__(self,responseTrain, classTrain, responseTest, num_neighbors=10):

        self.responseTrain = responseTrain
        self.responseTest = responseTest
        self.classTrain = classTrain
        self.classes = np.unique(classTrain)
        self.numClasses = len(self.classes)
        self.numNeighbors = num_neighbors

        self.defineClassIDs()

    def defineClassIDs(self):
        '''these classIDs are internal, not like the other two classifiers here!'''
        classIDs = {}
        for i,c in enumerate(self.classes):
            classIDs[c] = i

        self.classIDs = classIDs


    def fit(self, responseTest, responseTrain, metric='corr'):

        Ttest, N = self.responseTest.shape
        Ttrain, _ = self.responseTrain.shape

        distance = np.zeros((Ttest, Ttrain))
        for i in range(Ttest):
            for j in range(Ttrain):
                dist_temp = corr_dist(self.responseTest[i], self.responseTrain[j])
                if np.isnan(dist_temp):
                    distance[i, j] = 0.
                else:
                    distance[i, j] = dist_temp

        self.dist = distance

    def predict(self):

        # assume distance matrix has been calculated
        Ttest, N = self.responseTest.shape
        predictions = []
        for t in range(Ttest):

            neighbors = np.argsort(self.dist[t])[:self.numNeighbors]
            predictions.append(st.mode(self.classTrain[neighbors])[0][0])

        return np.array(predictions)


def circdist(ori1, ori2):
    '''
    calculate shortest distance around a circle between two orientations
    '''

    ori_diff = np.abs(np.fmod(ori1 - ori2, 360.))
    if ori_diff >= 180:
        ori_diff = 360. - ori_diff

    return ori_diff


def calc_distance_error(predictions, labels, stim_class, stim_category, stim_template=None):

    error_val = 0.
    if (stim_class == 'ns') or (stim_class == 'natural_scenes'):

        if stim_template is None:
            raise Exception('need stim template for natural scenes')

        for i, n in enumerate(predictions):
            error_val += np.sqrt(np.mean((stim_template[n] - stim_template[labels[i]])**2))
        error_val /= float(len(predictions))

    elif ('nm' in stim_class) or ('natural_movie' in stim_class):
        error_val = np.sqrt(np.mean( (predictions - labels)**2)) # rmse in frames

    elif (stim_class == 'sg') or (stim_class == 'static_gratings'):
        if (stim_category == 'orientation') or (stim_category == 'spatial_frequency'):
            count = 1
            for i, n in enumerate(predictions):
                if (n != '-1') and (labels[i] != '-1'): # discard blank
                    error_val += np.abs(float(n) - float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'all': # orientation, spatial frequency in that order for each stim

            count = 1
            for i, stim_predict in enumerate(predictions):
                if (stim_predict != '-1') and (labels[i] != '-1'): # discard blank

                    stim_predict = stim_predict[1:-1].split(' ')
                    stim_predict = [x for x in stim_predict if x != '']

                    stim_predict0 = float(stim_predict[0])
                    stim_predict1 = float(stim_predict[1])

                    label = labels[i][1:-1].split(' ')
                    label = [x for x in label if x != '']

                    label0 = float(label[0])
                    label1 = float(label[1])

                    error_val += np.sqrt( (stim_predict0 - label0)**2 + (stim_predict1 - label1)**2 )
                    count += 1

            error_val /= float(count)

    elif (stim_class == 'dg') or (stim_class == 'drifting_gratings'):

        if stim_category == 'orientation':
            count = 1
            for i, n in enumerate(predictions):
                if (n != '-1') and (labels[i] != '-1'): # discard blank
                    error_val += circdist(float(n), float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'temporal_frequency':
            count = 1
            for i, n in enumerate(predictions):
                if (n != -1) and (labels[i] != -1): # discard blank
                    error_val += np.abs(float(n) - float(labels[i])) # how many degrees or cycles/degree off (here degrees only between 0, 150)
                    count += 1

            error_val /= float(count)

        elif stim_category == 'all':

            count = 1
            for i, stim_predict in enumerate(predictions):
                if (stim_predict != '-1') and (labels[i] != '-1'): # discard blank

                    stim_predict = stim_predict[1:-1].split(' ')
                    stim_predict = [x for x in stim_predict if x != '']

                    stim_predict0 = float(stim_predict[0])
                    stim_predict1 = float(stim_predict[1])

                    label = labels[i][1:-1].split(' ')
                    label = [x for x in label if x != '']

                    label0 = float(label[0])
                    label1 = float(label[1])

                    error_val += np.sqrt( (stim_predict0 - label0)**2 + (stim_predict1 - label1)**2 )
                    count += 1

            error_val /= float(count)

    else:
        raise Exception('No relative error defined for stim class'+str(stim_class))

    return error_val


def get_response_table(dff, stim_table, stim, log=False, shift=0, width=None):

    stim_lengths = stim_table.end.values - stim_table.start.values
    Nstim = len(stim_table)
    Ncells = dff.shape[0]

    if width is None:  # use stim length - this should correspond to the SDK sweep response

        if stim == 'ns' or stim == 'natural_scenes':
            ind_start = shift
            ind_end = ind_start + np.amin(stim_lengths) + 7
        else:
            min_length = max([1, np.amin(stim_lengths)])
            ind_start = shift
            ind_end = ind_start + min_length

    else: # use pre-defined window width

        ind_start = shift
        ind_end = ind_start + width


    ''' get response array '''
    response_array = np.zeros((Ncells, Nstim))
    for i in range(Nstim):
        response_array[:, i] += np.mean(dff[:, stim_table.start.values[i]+ind_start:stim_table.start.values[i]+ind_end], axis=1)

    if log:
        response_array += 1. - np.amin(response_array)
        response_array = np.log(response_array)


    return response_array.T  # stim x cells


def get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, nan_ind=None, log=False, shift=0, width=14):
    '''
    get array of responses and running speeds throughout experiment
    '''


    N, T = dff.shape

    if master_stim_table is not None:
        start = master_stim_table['start'][0]
    else:
        start = 1000

    T -= start

    width = int(width)
    Nstim = np.floor(np.float(T) / np.float(width)).astype('int')

    response_array = np.zeros((Nstim, N))
    running_array = np.zeros((Nstim))

    dxcm = np.ma.masked_array(dxcm, mask=np.isnan(dxcm)).reshape(-1, 1)

    for i in range(Nstim):
        response_array[i] = np.mean(dff[:, start+(i*width):start+(i+1)*width], axis=1)
        running_array[i] = np.ma.mean(dxcm[start+(i*width):start+(i+1)*width])

    if pupil_size is not None and nan_ind is not None:

        # pupil_size = np.ma.masked_array(pupil_size, mask=np.isnan(pupil_size))
        pupil_array = np.zeros((Nstim))
        bad_ind = []

        for i in range(Nstim):
            # pupil_array[i] = np.ma.mean(pupil_size[start+(i*width):start+(i+1)*width])
            ind = range(start+i*width, start+(i+1)*width)
            if not np.all([j in nan_ind for j in ind]):
                pupil_array[i] = np.mean(pupil_size[ind])
            else:
                bad_ind.append(i)

        response_array = np.delete(response_array, bad_ind, 0)
        running_array = np.delete(running_array, bad_ind, 0)
        pupil_array = np.delete(pupil_array, bad_ind, 0)

        return response_array, running_array, pupil_array

    else:
        return response_array, running_array, None


def get_running_table(dxcm, stim_table, stim, shift=0, width=None):

    stim_lengths = stim_table.end.values - stim_table.start.values
    Nstim = len(stim_table)

    dxcm = np.ma.masked_array(dxcm, mask=np.isnan(dxcm))

    if width is None:  # use stim length - this should correspond to the SDK sweep response

        if stim == 'ns' or stim == 'natural_scenes':
            ind_start = shift
            ind_end = ind_start + np.amin(stim_lengths) + 7
        else:
            min_length = max([1, np.amin(stim_lengths)])
            ind_start = shift
            ind_end = ind_start + min_length

    else: # use pre-defined window width
        ind_start = shift
        ind_end = ind_start + width

    running_array = np.zeros((Nstim))

    for i in range(Nstim):
        start = stim_table.start.values[i] + ind_start
        end = stim_table.start.values[i] + ind_end
        running_array[i] = np.ma.mean(dxcm[start:end])

    return running_array


def interpolate_pupil_size(pupil_size):

    nan_mask = np.isnan(pupil_size)
    nan_ind = np.where(nan_mask)[0]
    good_ind = np.where(~nan_mask)[0]

    pupil_size[nan_ind] = np.interp(nan_ind, good_ind, pupil_size[good_ind])

    return pupil_size, nan_ind


def nested_cross_val_LDA_factors(response, stims, num_folds=5, plot=False, shrinkage='ledoit-wolf'):
    ''' nested cross-validation for number of factors to use for LDA with predicted covariances '''

    Nstim, N = response.shape
    num_factors_range = range(1, 10)

    lda_train_scores = np.zeros((len(num_factors_range), num_trials, num_folds))
    lda_validation_scores = np.zeros((len(num_factors_range), num_trials, num_folds))

    # best_model_num_factors = np.zeros((num_trials, num_folds))
    # lda_best_model_test_scores = np.zeros((num_trials, num_folds))
    # lda_best_model_train_scores = np.zeros((num_trials, num_folds))

    # for i in range(num_trials):
        # skf_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=i)

    skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # outer_fold = 0
    # for selection, test in skf_outer.split(response, stims):

    train_scores_temp = np.zeros((len(num_factors_range), num_folds))
    validation_scores_temp = np.zeros((len(num_factors_range), num_folds))

    for n, num_fact in enumerate(num_factors_range):

        inner_fold = 0
        for train, validation in skf_inner.split(response, stims):
            rTrain = response[train]
            cTrain = stims[train]
            rValid = response[validation]
            cValid = stims[validation]

            lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0, shrinkage=shrinkage)
            lda.fit(rTrain)
            train_predictions = lda.predict(rTrain)
            train_scores_temp[n, inner_fold] = np.sum((train_predictions == cTrain)) / float(len(train))

            lda.fit(rValid)
            validation_predictions = lda.predict(rValid)
            validation_scores_temp[n, inner_fold] = np.sum((validation_predictions == cValid)) / float(
                len(validation))

            inner_fold += 1

        lda_train_scores[n] = train_scores_temp[n].mean()
        lda_validation_scores[n] = validation_scores_temp[n].mean()

    mean_validation_scores = validation_scores_temp.mean(axis=1)
    ind_best = np.where(mean_validation_scores == max(mean_validation_scores))[0][0]
    # best_model_num_factors[i, outer_fold] = num_factors_range[ind_best]
    best_model_num_factors = num_factors_range[ind_best]

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[1].errorbar(num_factors_range, lda_validation_scores.mean(axis=2), yerr=lda_validation_scores.std(axis=2)/float(num_folds), label='validation')
        ax[0].errorbar(num_factors_range, lda_train_scores.mean(axis=2), yerr=lda_train_scores.std(axis=2)/float(num_folds), label='train')
        for i in range(num_folds):
            ax[0].plot(num_factors_range, lda_train_scores[:, 0, i], 'k', alpha=0.2)
            ax[1].plot(num_factors_range, lda_validation_scores[:, 0, i], 'k', alpha=0.2)

        # ax.legend(loc=0)
        # ax.set_xlabel('Number of factors')
        # ax.set_ylabel('Accuracy')
        ax[0].set_title('Train')
        ax[1].set_title('Validation')
        fig.text(.5, .05, 'Number of factors', ha='center')
        fig.text(.05, .5, 'Accuracy', va='center', rotation='vertical')


    return best_model_num_factors


def nested_cross_val_KNeighbors(response, stims, num_folds=3, plot=False):
    ''' nested cross-validation for number of factors to use for LDA with predicted covariances '''

    # skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True)
    skf_inner = KFold(n_splits=num_folds, shuffle=True)

    Nstim, N = response.shape
    # num_neighbors_range = range(5, min(100, Nstim*(num_folds-1)).astype('int'), 20)

    max_num_neighbors_fold = []
    for train, validation in skf_inner.split(response, stims):
        max_num_neighbors_fold.append(len(train))

    max_num_neighbors_fold = np.unique(max_num_neighbors_fold)

    # num_neighbors_range = range(5, min(max_num_neighbors_fold), 20)
    # num_neighbors_range = np.round(np.logspace(0., 2., num=8)).astype('int')
    num_neighbors_range = [1] + range(5, 100, 10)

    if max(num_neighbors_range) > min(max_num_neighbors_fold):
        ind = np.where(num_neighbors_range > min(max_num_neighbors_fold))[0][0]
        num_neighbors_range = num_neighbors_range[:ind]

    # print 'testing ' + str(len(num_neighbors_range)) + ' possible num_neighbors'
    train_scores = np.zeros((len(num_neighbors_range), num_folds))
    validation_scores = np.zeros((len(num_neighbors_range), num_folds))

    for n, num in enumerate(num_neighbors_range):

        # knn = KNeighborsClassifier(n_neighbors=num, weights='uniform', algorithm='ball_tree')

        inner_fold = 0
        for train, validation in skf_inner.split(response, stims):
            rTrain = response[train]
            cTrain = stims[train]
            rValid = response[validation]
            cValid = stims[validation]

            knn = KNN(rTrain, cTrain, rTrain, num_neighbors=num)
            knn.fit(rTrain, rTrain)
            train_predictions = knn.predict()

            knn = KNN(rTrain, cTrain, rValid, num_neighbors=num)
            knn.fit(rValid, rTrain)
            validation_predictions = knn.predict()

            train_scores[n, inner_fold] = np.sum((train_predictions == cTrain)) / float(len(train))
            validation_scores[n, inner_fold] = np.sum((validation_predictions == cValid)) / float(
                len(validation))

            inner_fold += 1


    mean_validation_scores = validation_scores.mean(axis=1)
    ind_best = np.where(mean_validation_scores == max(mean_validation_scores))[0][0]
    best_num_neighbors = num_neighbors_range[ind_best]

    return best_num_neighbors


def shuffle_trials_within_class(response, classes):
    '''
    shuffle trials across neurons within the same class to decorrelate them
    :param response: trials x neurons
    :param classes: trials
    :return:
    '''

    T, N = response.shape
    response_new = np.zeros((T, N))

    for i, c in enumerate(np.unique(classes)):

        ind = (classes == c)
        for n in range(N):
            response_new[ind, n] = np.random.permutation(response[ind, n])

    return response_new


def compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner=2):

    if (method in ['LDA', 'LDARun', 'LDAStat', 'LDADilate', 'LDAConstrict']) or ('ShuffledLDA' in method):
        lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif 'bootstrapLDA' in method:

        lda = gc.bootstrapLDA(rTrain, cTrain, num_factors=0, lam=0.1, numBoots=100, shrinkage='diagonal')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['diagLDA', 'diagLDARun', 'diagLDAStat', 'diagLDADilate', 'diagLDAConstrict']:
        lda = gc.LDA(rTrain, cTrain, num_factors=0, lam=1., shrinkage='diagonal')
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['LDAFactor', 'LDAFactorRun', 'LDAFactorStat', 'LDAFactorDilate', 'LDAFactorConstrict']:
        lda = gc.LDA(rTrain, cTrain, num_factors=1, lam=0.1, shrinkage=None)
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['NaiveBayes', 'NaiveBayesRun', 'NaiveBayesStat', 'NaiveBayesDilate', 'NaiveBayesConstrict']:
        nb = gc.NaiveBayes(rTrain, cTrain, lam=0.1)
        nb.fit(rTrain)
        train_predictions = nb.predict(rTrain)

        nb.fit(rTest)
        test_predictions = nb.predict(rTest)

        return train_predictions, test_predictions, None

    elif method in ['GDA', 'GDARun', 'GDAStat', 'GDADilate', 'GDAConstrict']:
        gda = gc.GDA(rTrain, cTrain, num_factors=0, lam=0.1, shrinkage='ledoit-wolf')
        gda.fit(rTrain)
        train_predictions = gda.predict(rTrain)

        gda.fit(rTest)
        test_predictions = gda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method == 'GDAFactor':

        gda = gc.GDA(rTrain, cTrain, num_factors=1, lam=0.1)
        gda.fit(rTrain)
        train_predictions = gda.predict(rTrain)

        gda.fit(rTest)
        test_predictions = gda.predict(rTest)

        return train_predictions, test_predictions, None

    elif method == 'LDABestFactor':

        num_fact = nested_cross_val_LDA_factors(rTrain, cTrain, num_folds=num_folds_inner, plot=False, shrinkage=None)
        lda = gc.LDA(rTrain, cTrain, num_factors=num_fact, lam=0.1)
        lda.fit(rTrain)
        train_predictions = lda.predict(rTrain)

        lda.fit(rTest)
        test_predictions = lda.predict(rTest)

        return train_predictions, test_predictions, num_fact

    elif 'KNeighbors' in method:

        num_neighb = nested_cross_val_KNeighbors(rTrain, cTrain, num_folds=num_folds_inner, plot=False)

        knn = KNN(rTrain, cTrain, rTrain, num_neighbors=num_neighb)
        knn.fit(rTrain, rTrain)
        train_predictions = knn.predict()

        knn = KNN(rTrain, cTrain, rTest, num_neighbors=num_neighb)
        knn.fit(rTest, rTrain)
        test_predictions = knn.predict()

        return train_predictions, test_predictions, num_neighb

    elif 'RandomForest' in method:

        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(rTrain, cTrain)

        train_predictions = rfc.predict(rTrain)
        test_predictions = rfc.predict(rTest)

        return train_predictions, test_predictions, None


def run_decoding_expt(expt, standardize=True, subsampleBehavior=True, flat_class=True, detect_events=True, save_dir='/local1/Documents/projects/cam_analysis/decode_results_L0events', analysis_file_dir='/allen/programs/braintv/workgroups/nc-ophys/ObservatoryPlatformPaperAnalysis/analysis_files_pre_2018_3_29'):

    '''
    :return:
    '''

    # print expt
    expt = int(expt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # decode_methods = ['diagLDA', 'LDA', 'ShuffledLDA']
    # decode_methods = ['KNeighbors']
    # suffixes=['Run','Stat']
    # decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'RandomForest', 'ShuffledKNeighbors', 'ShuffledLDA','ShuffledRandomForest'] # need to check memory scaling of random forests for decoding movies
    decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'KNeighbors', 'ShuffledKNeighbors', 'ShuffledLDA'] # need to check memory scaling of random forests for decoding movies

    # decode_methods = ['diagLDA', 'LDA', 'NaiveBayes', 'LDAFactor', 'bootstrapLDA', 'KNeighbors', 'ShuffledKNeighbors', 'ShuffledLDA']
    suffixes = ['', 'Stat', 'Run', 'Dilate', 'Constrict']

    methods = []
    for method in decode_methods:
        for suffix in suffixes:
            methods.append(method+suffix)

    stim_class_dict = {'A': ['dg'], 'B': ['sg','ns'], 'C': [], 'C2': []} # not decoding movies
    # stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}
    stim_category_dict = {'dg':['orientation','temporal_frequency'], 'sg':['orientation','spatial_frequency'],'ns':['frame'], 'nm1':['frame'], 'nm2':['frame'], 'nm3':['frame']}


    run_thresh = 1.
    min_stim_repeats = 5 # must be at least num_folds * num_folds of inner cross-val for stratified k-fold; for k-fold will have at least 2*min_repeats points
    num_folds = 5
    num_folds_inner = 2

    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)

    try:
        expt_stimuli = data_set.list_stimuli()
    except:
        raise Exception('no data for exp '+str(expt)+', skipping')
        # continue

    if 'drifting_gratings' in expt_stimuli:
        session_type = 'A'
    elif 'static_gratings' in expt_stimuli:
        session_type = 'B'
    elif 'locally_sparse_noise' in expt_stimuli:
        session_type = 'C'
    elif 'locally_sparse_noise_8deg' in expt_stimuli:
        session_type = 'C2'
    else:
        raise Exception('Session type not defined')

    if session_type == 'A':
        run_window = 60
    else:
        run_window = 15

    test_dict = {}
    train_dict = {}
    test_shuffle_dict = {}
    train_shuffle_dict = {}

    test_dist_dict = {}
    train_dist_dict = {}
    test_shuffle_dist_dict = {}
    train_shuffle_dist_dict = {}

    num_factors_dict = {}
    num_factors_shuffle_dict = {}
    num_neighbors_dict = {}
    num_neighbors_shuffle_dict = {}
    test_confusion_dict = {}
    train_confusion_dict = {}

    _, dff = data_set.get_dff_traces()
    dxcm, _ = data_set.get_running_speed()

    N, T = dff.shape
    try:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
    except:
        print('no pupil size information')
        pupil_size = None

    if detect_events:

        l0 = L0_analysis(data_set)
        dff = l0.get_events()

        # event_file = os.path.join(event_dir, 'expt_'+str(expt)+'_events.npy')
        # dff = np.load(event_file)
        #
        # noise_stds = np.load(os.path.join(event_dir, 'expt_'+str(expt)+'_NoiseStd.npy'))[:, None]
        # dff *= noise_stds * 10. # rescale to units of df/f

    print('decoding running/stationary')
    methods_calc = decode_methods
    try:
        master_stim_table = data_set.get_stimulus_table('master')
    except:
        print('no good master stim table')
        master_stim_table = None

    if pupil_size is not None:
        pupil_t, pupil_size = data_set.get_pupil_size()
        pupil_size, nan_ind = interpolate_pupil_size(pupil_size)
        response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind, width=run_window)
    else:
        print('no pupil size information')
        response, running_speed, _ = get_tables_exp(master_stim_table, dff, dxcm, pupil_size=None, width=run_window)
        pupil_array = None
        methods = [m for m in methods if ('Dilate' not in m) and ('Constrict' not in m) ]


    # fit mixture model to running speeds to classify running / stationary
    good_ind = np.isfinite(running_speed)
    # X = np.array(running_speed[good_ind]).reshape(-1, 1)
    X = np.array(running_speed[good_ind]).reshape(-1, 1)

    run_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.2, covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)
    run_dpgmm.fit(X)
    Y = run_dpgmm.predict(X)
    means = run_dpgmm.means_
    vars = run_dpgmm.covariances_

    labels = -1*np.ones(running_speed.shape, dtype=np.int)
    labels[good_ind] = Y

    run_states = np.unique(Y)
    means = [means[i][0] for i in run_states]
    vars = [vars[i][0] for i in run_states]

    stationary_label = [run_states[i] for i in range(len(run_states)) if means[i] < run_thresh]
    stationary_vars = [vars[i] for i in range(len(run_states)) if means[i] < run_thresh]

    Nstationary_labels = len(stationary_label)
    stationary_label = [stationary_label[i] for i in range(Nstationary_labels) if
                        stationary_vars[i] is min(stationary_vars)]

    # running stuff
    if Nstationary_labels != 0:
        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == stationary_label).astype('float')  # thing to decode!
        stim_types = np.unique(stims_calc)
    else:
        stim_types = [-10]

    if len(stim_types) > 1:
        if subsampleBehavior and flat_class:
            NperStat = sum(stims_calc == 1.)
            NperRun = len(stims_calc) - NperStat
            Nsweeps = min(NperStat, NperRun)

            RunSweeps = np.random.choice(np.where(stims_calc == 0.)[0], size=Nsweeps)
            StatSweeps = np.random.choice(np.where(stims_calc == 1.)[0], size=Nsweeps)
            SweepInds = np.concatenate((RunSweeps, StatSweeps), axis=0)
            stims_calc = stims_calc[SweepInds]
            response = response[SweepInds]

        stims_shuffle_calc = np.random.permutation(stims_calc)

        if len(response) >= num_folds:
            for method in methods_calc:

                print(method)

                skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

                if 'Shuffled' in method:
                    response_calc = shuffle_trials_within_class(response, stims_calc)
                else:
                    response_calc = response.copy()

                test_scores = np.zeros((num_folds))
                train_scores = np.zeros((num_folds))
                train_shuffle_scores = np.zeros((num_folds))
                test_shuffle_scores = np.zeros((num_folds))

                # test_confusion = np.zeros((len(stim_types), len(stim_types)))
                # train_confusion = np.zeros((len(stim_types), len(stim_types)))

                if 'LDABestFactor' in method:
                    num_factors = np.zeros((num_folds))
                    num_factors_shuffle = np.zeros((num_folds))

                if 'KNeighbors' in method:
                    num_neighbors = np.zeros((num_folds))
                    num_neighbors_shuffle = np.zeros((num_folds))

                fold = 0
                scaler = StandardScaler()
                for train, test in skf.split(response_calc, stims_calc):

                    rTrain = response_calc[train]
                    rTest = response_calc[test]
                    cTrain = stims_calc[train]
                    cTest = stims_calc[test]

                    if standardize:
                        scaler.fit(rTrain)
                        rTrain = scaler.transform(rTrain)
                        rTest = scaler.transform(rTest)

                    train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                    if method == 'KNeighbors':
                        num_neighbors[fold] = param
                    elif method == 'LDABestFactor':
                        num_factors[fold] = param

                    train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                    test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                    fold += 1

                    # if len(np.unique(cTrain)) < len(stim_types)
                    #     # get labels for train stims, expand temp confusion to full size
                    #     temp_confusion1 = confusion_matrix(cTrain, train_predictions)
                    #     cTrain_labels = np.unique(cTrain)
                    #     temp_confusion2 = np.zeros((len(stim_types), len(stim_types)))
                    #     for ss, stim_temp in enumerate(stim_types):
                    #         ind = np.where()
                    #
                    # else:

                    # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                    # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

                fold = 0
                for train, test in skf.split(response_calc, stims_shuffle_calc):

                    rTrain = response_calc[train]
                    rTest = response_calc[test]
                    cTrain = stims_shuffle_calc[train]
                    cTest = stims_shuffle_calc[test]

                    if standardize:
                        scaler.fit(rTrain)
                        rTrain = scaler.transform(rTrain)
                        rTest = scaler.transform(rTest)

                    train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                    if method == 'KNeighbors':
                        num_neighbors[fold] = param
                    elif method == 'LDABestFactor':
                        num_factors[fold] = param

                    train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                    test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                    fold += 1

                test_dict['runState' + '_' + method] = test_scores
                train_dict['runState' + '_' + method] = train_scores
                test_shuffle_dict['runState' + '_' + method] = test_shuffle_scores
                train_shuffle_dict['runState' + '_' + method] = train_shuffle_scores

                # test_confusion_dict[stim_class + '_' + stim_category + '_' + method] = test_confusion
                # train_confusion_dict[stim_class + '_' + stim_category + '_' + method] = train_confusion

                if method == 'LDABestFactor':
                    num_factors_dict['runState'] = num_factors
                    num_factors_shuffle_dict['runState'] = num_factors_shuffle

                elif 'KNeighbors' in method:
                    num_neighbors_dict['runState' + '_' + method] = num_neighbors
                    num_neighbors_shuffle_dict['runState' + '_' + method] = num_neighbors_shuffle

    # decode pupil diameter - wide / narrow
    if (pupil_array is not None):
        print('decoding pupil wide / narrow')

        response, running_speed, pupil_array = get_tables_exp(master_stim_table, dff, dxcm, pupil_size, nan_ind, width=run_window) # get full response array again

        pupil_dpgmm = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.2,
                                                    covariance_type='diag', max_iter=2000, tol=1e-3, n_init=1)

        good_ind = np.isfinite(pupil_array)
        X = pupil_array[good_ind].reshape(-1, 1)

        pupil_dpgmm.fit(X)
        Y = pupil_dpgmm.predict(X)
        means = pupil_dpgmm.means_
        vars = pupil_dpgmm.covariances_

        labels = -1 * np.ones(pupil_array.shape, dtype=np.int)
        labels[good_ind] = Y

        dilate_states = np.unique(Y)
        means = [means[i][0] for i in dilate_states]

        constrict_label = dilate_states[np.where(means == np.amin(means))]

        ''' get mean response arrays during running and during stationary and all activity '''
        stims_calc = (labels == constrict_label).astype('float')  # thing to decode!
        stims_shuffle_calc = np.random.permutation(stims_calc)

        stim_types = np.unique(stims_calc)

        if subsampleBehavior and flat_class:
            NperConstrict = sum(stims_calc == 1.)
            NperDilate = len(stims_calc) - NperConstrict
            Nsweeps = min(NperConstrict, NperDilate)

            DilateSweeps = np.random.choice(np.where(stims_calc == 0.)[0], size=Nsweeps)
            ConstrictSweeps = np.random.choice(np.where(stims_calc == 1.)[0], size=Nsweeps)
            SweepInds = np.concatenate((DilateSweeps, ConstrictSweeps), axis=0)
            stims_calc = stims_calc[SweepInds]
            response = response[SweepInds]

        stims_shuffle_calc = np.random.permutation(stims_calc)

    if (len(stim_types) > 1) and (len(response) >= num_folds):
        for method in methods_calc:

            print(method)

            skf = StratifiedKFold(n_splits=num_folds, shuffle=False)

            if 'Shuffled' in method:
                response_calc = shuffle_trials_within_class(response_calc, stims_calc)
            else:
                response_calc = response.copy()

            test_scores = np.zeros((num_folds))
            train_scores = np.zeros((num_folds))
            train_shuffle_scores = np.zeros((num_folds))
            test_shuffle_scores = np.zeros((num_folds))

            # test_confusion = np.zeros((len(stim_types), len(stim_types)))
            # train_confusion = np.zeros((len(stim_types), len(stim_types)))

            if 'LDABestFactor' in method:
                num_factors = np.zeros((num_folds))
                num_factors_shuffle = np.zeros((num_folds))

            if 'KNeighbors' in method:
                num_neighbors = np.zeros((num_folds))
                num_neighbors_shuffle = np.zeros((num_folds))

            fold = 0

            scaler = StandardScaler()
            for train, test in skf.split(response_calc, stims_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_calc[train]
                cTest = stims_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                fold += 1

                # train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                # test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)

            fold = 0
            for train, test in skf.split(response_calc, stims_shuffle_calc):

                rTrain = response_calc[train]
                rTest = response_calc[test]
                cTrain = stims_shuffle_calc[train]
                cTest = stims_shuffle_calc[test]

                if standardize:
                    scaler.fit(rTrain)
                    rTrain = scaler.transform(rTrain)
                    rTest = scaler.transform(rTest)

                train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                if method == 'KNeighbors':
                    num_neighbors[fold] = param
                elif method == 'LDABestFactor':
                    num_factors[fold] = param

                train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))
                fold += 1

            test_dict['pupil' + '_' + method] = test_scores
            train_dict['pupil' + '_' + method] = train_scores
            test_shuffle_dict['pupil' + '_' + method] = test_shuffle_scores
            train_shuffle_dict['pupil' + '_' + method] = train_shuffle_scores

            if method == 'LDABestFactor':
                num_factors_dict['pupil'] = num_factors
                num_factors_shuffle_dict['pupil'] = num_factors_shuffle

            elif 'KNeighbors' in method:
                num_neighbors_dict['pupil' + '_' + method] = num_neighbors
                num_neighbors_shuffle_dict['pupil' + '_' + method] = num_neighbors_shuffle

    print('decoding visual stimulus')
    stim_class_list = stim_class_dict[session_type]
    if len(stim_class_list) > 0:         # decode visual stimulus
        for stim_class in stim_class_list:
            if stim_class == 'ns':
                stim_table = data_set.get_stimulus_table('natural_scenes')
                stim_template = data_set.get_stimulus_template('natural_scenes')
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'NaturalScenes', str(expt)+'_ns_events_analysis.h5'))

            elif stim_class == 'dg':
                stim_table = data_set.get_stimulus_table('drifting_gratings')
                stim_template = None
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'DriftingGratings', str(expt)+'_dg_events_analysis.h5'))

            elif stim_class == 'sg':
                stim_table = data_set.get_stimulus_table('static_gratings')
                stim_template = None
                analysis_file = h5py.File(os.path.join(analysis_file_dir, 'StaticGratings', str(expt)+'_sg_events_analysis.h5'))

            # elif stim_class == 'nm1':
            #     stim_table = data_set.get_stimulus_table('natural_movie_one')
            #     stim_template = data_set.get_stimulus_template('natural_movie_one')
            #
            # elif stim_class == 'nm2':
            #     stim_table = data_set.get_stimulus_table('natural_movie_two')
            #     stim_template = data_set.get_stimulus_template('natural_movie_two')
            #
            # elif stim_class == 'nm3':
            #     stim_table = data_set.get_stimulus_table('natural_movie_three')
            #     stim_template = data_set.get_stimulus_template('natural_movie_three')

            response = analysis_file['mean_sweep_events'].values()[3].value # trials x neurons

            if 'nm' in stim_class:
                # response = get_response_table(dff, stim_table, stim=stim_class, log=False, width=6)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class, width=6)
            else:
                # response = get_response_table(dff, stim_table, stim=stim_class, log=False)  # trials x neurons
                running_speed = get_running_table(dxcm, stim_table, stim=stim_class)

            labels = run_dpgmm.predict(running_speed.reshape(-1, 1))
            ind_stationary = (labels == stationary_label)
            ind_run = ~ind_stationary

            try:
                pupil_t, pupil_size = data_set.get_pupil_size()
                pupil_size, nan_ind = interpolate_pupil_size(pupil_size)

                if 'nm' in stim_class:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class, width=6)
                else:
                    pupil_array = get_running_table(pupil_size, stim_table, stim=stim_class)
                labels = pupil_dpgmm.predict(pupil_array.reshape(-1, 1))
                ind_constrict = (labels == constrict_label)

            except:
                print('no pupil size information')
                pupil_size = None

            ''' set up stim labels '''
            decode_list = list(stim_category_dict[stim_class])
            if len(decode_list) > 0:  print("\tStimulus class:  ", stim_class)
            if len(decode_list) > 1:  decode_list.append('all')

            for stim_category in decode_list:
                print("\t\tStimulus Category:  ", stim_category)
                if (stim_category != 'all') and (stim_category != 'run'):
                    stims = np.array(stim_table[stim_category])
                elif stim_category == 'run':
                    stims = ind_stationary.astype('int')
                else:
                    stim_temp_array = [np.array(stim_table[stim_temp]) for stim_temp in decode_list[:-1]]
                    stims = np.vstack(stim_temp_array).T
                    stims = np.array([str(s_str) for s_str in stims])

                if stim_class == 'dg':

                    if stim_category == 'temporal_frequency':
                        stims = np.ceil(stims*100)
                    blank_mask = np.array(stim_table['blank_sweep'].astype('bool'))
                    stims[blank_mask] = -1

                elif stim_class == 'sg':

                    if stim_category == 'spatial_frequency':
                        stims = np.ceil(stims*100)

                    if stim_category != 'all':
                        blank_mask = ~np.isfinite(stims)
                        stims[blank_mask] = -1
                    else:
                        blank_mask = np.zeros(stims.shape).astype('bool')
                        for kk in range(len(stims)):
                            if 'nan' in stims[kk]: blank_mask[kk] = True
                        stims[blank_mask] = -1


                ''' set up response / stims for running '''

                stims_shuffle = np.random.permutation(stims)
                stim_types = np.unique(stims)


                if Nstationary_labels > 0:
                    response_stationary = response[ind_stationary, :]
                    response_run = response[ind_run, :]

                    stims_stationary = stims[ind_stationary]
                    stims_run = stims[ind_run]

                    stim_types_stat = np.unique(stims_stationary)
                    stim_types_run = np.unique(stims_run)

                    Nperstim_stat = [sum(stims_stationary == c) for c in stim_types_stat]
                    Nperstim_run = [sum(stims_run == c) for c in stim_types_run]


                    ''' for running / stationary, only use stims that have >= min_stim_repeats repetitions '''
                    methods_calc = methods[:] # slice to copy

                    # if len(Nperstim_run) > 1:
                    #     stims_run_good = [stim for i, stim in enumerate(stim_types_run) if Nperstim_run[i] >= min_stim_repeats]
                    #     Nperstim_run = [i for i in Nperstim_run if i >= min_stim_repeats]
                    #
                    # if len(Nperstim_stat) > 1:
                    #     stims_stat_good = [stim for i, stim in enumerate(stim_types_stat) if Nperstim_stat[i] >= min_stim_repeats]
                    #     Nperstim_stat = [i for i in Nperstim_stat if i >= min_stim_repeats]

                    if (len(Nperstim_run) > 1) and (len(Nperstim_stat) > 1):
                        stims_run_good = [stim for i, stim in enumerate(stim_types_run) if (Nperstim_run[i] >= min_stim_repeats)]
                        stims_stat_good = [stim for i, stim in enumerate(stim_types_stat) if (Nperstim_stat[i] >= min_stim_repeats)]

                        stims_run_good = [stim for i, stim in enumerate(stims_run_good) if stim in stims_stat_good]
                        stims_stat_good = stims_run_good

                    if (len(Nperstim_run) <= 1) or (len(Nperstim_stat) <= 1) or (len(stims_run_good) < 2) or (len(stims_stat_good) < 2):
                        print('not enough stims during running and stationary')
                        methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]
                        stims_run_good = []
                        stims_stat_good = []

                    if (len(stims_run_good) > 0) and (len(stims_stat_good) > 0):
                        if subsampleBehavior: # use same total number of trials per stimulus in each condition

                            if flat_class:
                                stim_types_run = np.unique(stims_run_good)  # already know has to be in stims_stat good

                                stims_stat_mask = [False for x in range(len(stims_stationary))]
                                stims_run_mask = [False for x in range(len(stims_run))]

                                Nperstim_list = []
                                for stim in stim_types_run:
                                    Nperstim_list.append(min([sum(stims_stationary == stim), sum(stims_run == stim)]))

                                Nperstim = min(Nperstim_list)

                                for stim in stim_types_run:
                                    stim_ind_stat = np.random.choice(np.where(stims_stationary == stim)[0], size=Nperstim,
                                                                     replace=False)
                                    stim_ind_run = np.random.choice(np.where(stims_run == stim)[0], size=Nperstim,
                                                                    replace=False)

                                    for i in stim_ind_stat: stims_stat_mask[i] = True
                                    for i in stim_ind_run: stims_run_mask[i] = True


                            else:
                                stim_types_run = np.unique(stims_run_good) # already know has to be in stims_stat good

                                stims_stat_mask = [False for x in range(len(stims_stationary))]
                                stims_run_mask = [False for x in range(len(stims_run))]

                                for stim in stim_types_run:

                                    Nperstim = min([sum(stims_stationary == stim), sum(stims_run == stim)])

                                    stim_ind_stat = np.random.choice(np.where(stims_stationary == stim)[0], size=Nperstim, replace=False)
                                    stim_ind_run = np.random.choice(np.where(stims_run == stim)[0], size=Nperstim, replace=False)

                                    for i in stim_ind_stat: stims_stat_mask[i] = True
                                    for i in stim_ind_run: stims_run_mask[i] = True

                        else:
                            stims_stat_mask = [False for x in range(len(stims_stationary))]
                            for i, stim in enumerate(stims_stationary):
                                if stim in stims_stat_good:
                                    stims_stat_mask[i] = True

                            stims_run_mask = [False for x in range(len(stims_run))]
                            for i, stim in enumerate(stims_run):
                                if stim in stims_run_good:
                                    stims_run_mask[i] = True

                        stims_stat_mask = np.array(stims_stat_mask).astype('bool')
                        stims_run_mask = np.array(stims_run_mask).astype('bool')
                        stims_stationary = stims_stationary[stims_stat_mask]
                        response_stationary = response_stationary[stims_stat_mask, :]

                        stims_run = stims_run[stims_run_mask]
                        response_run = response_run[stims_run_mask, :]

                        stims_stationary_shuffle = np.random.permutation(stims_stationary)
                        stims_run_shuffle = np.random.permutation(stims_run)

                    else:
                        methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]
                else:
                    methods_calc = [m for m in methods_calc if ('Run' not in m) and ('Stat' not in m)]


                ''' set up response / stims for pupil narrow / wide '''
                if pupil_size is not None:

                    response_constrict = response[ind_constrict, :]
                    response_dilate = response[~ind_constrict, :]

                    stims_shuffle = np.random.permutation(stims)

                    stims_constrict = stims[ind_constrict]
                    stims_dilate = stims[~ind_constrict]

                    stim_types = np.unique(stims)
                    stim_types_constrict = np.unique(stims_constrict)
                    stim_types_dilate = np.unique(stims_dilate)

                    Nperstim_constrict = [sum(stims_constrict == c) for c in stim_types_constrict]
                    Nperstim_dilate = [sum(stims_dilate == c) for c in stim_types_dilate]

                    ''' for dilated/constricted, only use stims that have >= min_stim_repeats repetitions '''

                    if (len(Nperstim_dilate) > 1) and (len(Nperstim_constrict) > 1):
                        stims_dilate_good = [stim for i, stim in enumerate(stim_types_dilate) if
                                          (Nperstim_dilate[i] >= min_stim_repeats)]
                        stims_constrict_good = [stim for i, stim in enumerate(stim_types_constrict) if
                                           (Nperstim_constrict[i] >= min_stim_repeats)]

                        stims_dilate_good = [stim for i, stim in enumerate(stims_dilate_good) if stim in stims_constrict_good]
                        stims_constrict_good = stims_dilate_good


                    if (len(Nperstim_dilate) <= 1) or (len(Nperstim_constrict) <= 1) or (len(stims_dilate_good) < 2) or (len(stims_constrict_good) < 2):

                        print('not enough stims during dilated and constricted pupils')
                        methods_calc = [m for m in methods_calc if ('Constrict' not in m) and ('Dilate' not in m)]
                        stims_dilate_good = []
                        stims_constrict_good = []

                    if (len(stims_dilate_good) > 0) and (len(stims_constrict_good) > 0):
                        if subsampleBehavior:  # use same total number of trials per stimulus in each condition

                            if flat_class:
                                if flat_class:
                                    stim_types_dilate = np.unique(stims_dilate_good)  # already know has to be in stims_constrict good

                                    stims_constrict_mask = [False for x in range(len(stims_constrict))]
                                    stims_dilate_mask = [False for x in range(len(stims_dilate))]

                                    Nperstim_list = []
                                    for stim in stim_types_dilate:
                                        Nperstim_list.append(min([sum(stims_constrict == stim), sum(stims_dilate == stim)]))

                                    Nperstim = min(Nperstim_list)

                                    for stim in stim_types_dilate:
                                        stim_ind_constrict = np.random.choice(np.where(stims_constrict == stim)[0],
                                                                         size=Nperstim,
                                                                         replace=False)
                                        stim_ind_dilate = np.random.choice(np.where(stims_dilate == stim)[0], size=Nperstim,
                                                                        replace=False)

                                        for i in stim_ind_constrict: stims_constrict_mask[i] = True
                                        for i in stim_ind_dilate: stims_dilate_mask[i] = True



                            else:
                                stim_types_dilate = np.unique(stims_dilate_good)  # already know has to be in stims_constrict good

                                stims_constrict_mask = [False for x in range(len(stims_constrict))]
                                stims_dilate_mask = [False for x in range(len(stims_dilate))]

                                for stim in stim_types_dilate:

                                    Nperstim = min([sum(stims_constrict == stim), sum(stims_dilate == stim)])

                                    stim_ind_constrict = np.random.choice(np.where(stims_constrict == stim)[0], size=Nperstim,
                                                                     replace=False)
                                    stim_ind_dilate = np.random.choice(np.where(stims_dilate == stim)[0], size=Nperstim,
                                                                    replace=False)

                                    for i in stim_ind_constrict: stims_constrict_mask[i] = True
                                    for i in stim_ind_dilate: stims_dilate_mask[i] = True


                        else:
                            stims_constrict_mask = [False for x in range(len(stims_constrict))]
                            for i, stim in enumerate(stims_constrict):
                                if stim in stims_constrict_good:
                                    stims_constrict_mask[i] = True

                            stims_dilate_mask = [False for x in range(len(stims_dilate))]
                            for i, stim in enumerate(stims_dilate):
                                if stim in stims_dilate_good:
                                    stims_dilate_mask[i] = True

                        stims_constrict_mask = np.array(stims_constrict_mask).astype('bool')
                        stims_dilate_mask = np.array(stims_dilate_mask).astype('bool')
                        stims_constrict = stims_constrict[stims_constrict_mask]
                        response_constrict = response_constrict[stims_constrict_mask, :]

                        stims_dilate = stims_dilate[stims_dilate_mask]
                        response_dilate = response_dilate[stims_dilate_mask, :]

                        stims_constrict_shuffle = np.random.permutation(stims_constrict)
                        stims_dilate_shuffle = np.random.permutation(stims_dilate)

                else:
                    methods_calc = [m for m in methods_calc if ('Constrict' not in m) and ('Dilate' not in m)]


                for method in methods_calc:

                    print(method)

                    if 'Run' in method:
                        response_calc = response_run.copy()
                        stims_calc = stims_run.copy()
                        stims_shuffle_calc = stims_run_shuffle.copy()
                    elif 'Stat' in method:
                        response_calc = response_stationary.copy()
                        stims_calc = stims_stationary.copy()
                        stims_shuffle_calc = stims_stationary_shuffle.copy()
                    elif 'Dilate' in method:
                        response_calc = response_dilate.copy()
                        stims_calc = stims_dilate.copy()
                        stims_shuffle_calc = stims_dilate_shuffle.copy()
                    elif 'Constrict' in method:
                        response_calc = response_constrict.copy()
                        stims_calc = stims_constrict.copy()
                        stims_shuffle_calc = stims_constrict_shuffle.copy()
                    else:
                        response_calc = response.copy()
                        stims_calc = stims.copy()
                        stims_shuffle_calc = stims_shuffle.copy()

                    # skf = StratifiedKFold(n_splits=num_folds, shuffle=False)
                    skf = KFold(n_splits=num_folds, shuffle=False)

                    test_scores = np.zeros((num_folds))
                    train_scores = np.zeros((num_folds))
                    train_shuffle_scores = np.zeros((num_folds))
                    test_shuffle_scores = np.zeros((num_folds))

                    test_scores_dist = np.zeros((num_folds))
                    train_scores_dist = np.zeros((num_folds))
                    train_shuffle_scores_dist = np.zeros((num_folds))
                    test_shuffle_scores_dist = np.zeros((num_folds))

                    test_confusion = np.zeros((len(stim_types), len(stim_types)))
                    train_confusion = np.zeros((len(stim_types), len(stim_types)))

                    if 'LDABestFactor' in method:
                        num_factors = np.zeros((num_folds))
                        num_factors_shuffle = np.zeros((num_folds))

                    if 'KNeighbors' in method:
                        num_neighbors = np.zeros((num_folds))
                        num_neighbors_shuffle = np.zeros((num_folds))

                    fold = 0
                    scaler = StandardScaler()
                    for train, test in skf.split(response_calc, stims_calc):

                        rTrain = response_calc[train]
                        rTest = response_calc[test]
                        cTrain = stims_calc[train]
                        cTest = stims_calc[test]

                        if standardize:
                            scaler.fit(rTrain)
                            rTrain = scaler.transform(rTrain)
                            rTest = scaler.transform(rTest)

                        if 'Shuffled' in method:
                            rTest = shuffle_trials_within_class(rTest, cTest)

                        train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                        if method == 'KNeighbors':
                            num_neighbors[fold] = param
                        elif method == 'LDABestFactor':
                            num_factors[fold] = param

                        train_scores_dist[fold] = calc_distance_error(predictions=train_predictions, labels=cTrain, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)
                        test_scores_dist[fold] = calc_distance_error(predictions=test_predictions, labels=cTest, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)

                        train_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                        test_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                        fold += 1

                        train_confusion += confusion_matrix(cTrain, train_predictions, labels=stim_types)
                        test_confusion += confusion_matrix(cTest, test_predictions, labels=stim_types)


                    fold = 0
                    for train, test in skf.split(response_calc, stims_shuffle_calc):

                        rTrain = response_calc[train]
                        rTest = response_calc[test]
                        cTrain = stims_shuffle_calc[train]
                        cTest = stims_shuffle_calc[test]

                        if standardize:
                            scaler.fit(rTrain)
                            rTrain = scaler.transform(rTrain)
                            rTest = scaler.transform(rTest)

                        if 'Shuffled' in method:
                            rTest = shuffle_trials_within_class(rTest, cTest)

                        train_predictions, test_predictions, param = compute_prediction(rTrain, cTrain, rTest, method, num_folds_inner)
                        if method == 'KNeighbors':
                            num_neighbors_shuffle[fold] = param
                        elif method == 'LDABestFactor':
                            num_factors_shuffle[fold] = param

                        train_shuffle_scores_dist[fold] = calc_distance_error(predictions=train_predictions, labels=cTrain, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)
                        test_shuffle_scores_dist[fold] = calc_distance_error(predictions=test_predictions, labels=cTest, stim_class=stim_class, stim_category=stim_category, stim_template=stim_template)

                        train_shuffle_scores[fold] = np.sum(train_predictions == cTrain) / float(len(train))
                        test_shuffle_scores[fold] = np.sum(test_predictions == cTest) / float(len(test))

                        fold += 1

                    test_dict[stim_class + '_' + stim_category + '_' + method] = test_scores
                    train_dict[stim_class + '_' + stim_category + '_' + method] = train_scores
                    test_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores
                    train_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores

                    test_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_scores_dist
                    train_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_scores_dist
                    test_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = test_shuffle_scores_dist
                    train_shuffle_dist_dict[stim_class + '_' + stim_category + '_' + method] = train_shuffle_scores_dist

                    test_confusion_dict[stim_class + '_' + stim_category + '_' + method] = test_confusion
                    train_confusion_dict[stim_class + '_' + stim_category + '_' + method] = train_confusion

                    if method == 'LDABestFactor':
                        num_factors_dict[stim_class + '_' + stim_category] = num_factors
                        num_factors_shuffle_dict[stim_class + '_' + stim_category] = num_factors_shuffle

                    elif 'KNeighbors' in method:
                        num_neighbors_dict[stim_class + '_' + stim_category + '_' + method] = num_neighbors
                        num_neighbors_shuffle_dict[stim_class + '_' + stim_category + '_' + method] = num_neighbors_shuffle

                    # print 'train score: ' + str(np.mean(train_scores)) + ' (+/- ' +str(np.std(train_scores)*2) + ')'
                    # print 'test score: ' + str(np.mean(test_scores)) + ' (+/- ' +str(np.std(test_scores)*2) + ')'
                    # print 'shuffle score: ' + str(np.mean(test_shuffle_scores)) + ' (+/- ' +str(np.std(test_shuffle_scores)*2) + ')' #   # decode visual stimulus


    savefile = os.path.join(save_dir, str(expt)+'_test_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_error.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_train_dist.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_shuffle_dist_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_factors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_factors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_shuffle_num_neighbors.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(num_neighbors_shuffle_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_test_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(test_confusion_dict, error_file)
    error_file.close()

    savefile = os.path.join(save_dir, str(expt)+'_train_confusion.pkl')
    error_file = open(savefile, 'wb')
    pickle.dump(train_confusion_dict, error_file)
    error_file.close()

    # print 'session type: '+str(session_type)+' , elapsed time: '+str(time.time()-start_time)
