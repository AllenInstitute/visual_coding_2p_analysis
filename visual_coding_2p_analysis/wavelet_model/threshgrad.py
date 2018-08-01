import numpy as np
import itertools

def soft_rect(y, k=10):
    return np.maximum(1e-6, np.log(1. + np.exp(k * y)) / k)


def soft_rect_poisson(y, yhat, k=10, weights=None):
    yhat_nl = soft_rect(yhat, k)
    expka = np.exp(k * yhat)
    infidx = np.isinf(expka)
    expka = expka / (1. + expka)
    expka[infidx] = 1.
    grad, err = ((y / yhat_nl) - 1) * expka, -(y*np.log(yhat_nl) - yhat_nl)
    if weights is not None:
        grad, err = weights*grad, weights*err
    return grad, err


def lsq(y, yhat, weights=None):
    grad, err = y - yhat, .5*(y - yhat)**2
    if weights is not None:
        grad, err = weights*grad, weights*err
    return grad, err


def threshold_grad_desc_cv(X, y, delays, folds=6, step=.0001, testsize=.2, maxiter=150, threshold=.8,
                            adapt_up=1.2, adapt_down=.5, adapt_min=.0001, adapt_max=10000, err_function=lsq, weights=None, segment_size=None):

    """Uses threshold gradient descent to find a linear transformation of [X] with time [delays] that approximates [y].
    This procedure is repeated for [folds] in which each parameter in [threshold] is tested on a randomly
    sampled fraction of the data specified by [testsize]. The best model weights from each fold are averaged together
    and returned as the regression weights, i.e. bagged (bootstrap aggregation).

    Parameters
    ----------
    X : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    y : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
    delays : list or array_like, shape (D,)
        The time delays to include in the model. range(5) works fine for voxels.
    folds : integer (default: 10)
        The number of times to repeat the weight estimation.
    step : float in [0..1] (default: .0001)
        The step size to use in the gradient descent.
    testsize : float in [0..1] (default: .2)
        The fraction of the data to use for early stopping and choosing the threshold in each fold
    maxiter : integer (default: 150)
        The maximum number of iterations to run the gradient descent
    threshold : list or array_like, shape (T,) (default: [.3,.4,.5,.6,.7,.8,.9])
        The values of the threshold to test.
    adapt_up : float > 1. (default: 1.2)
        The step size is adaptively made larger by multiplying by this value when error on the test set decreases
    adapt_down : float < 1. (default: .5)
        The step size is adaptively made smaller by multiplying by this value when error on the test set increases
    adapt_min : float (default: .0001)
        The floor on the range of adaptation
    adapt_max : float (default: 10000)
        The ceiling on the range of adaptation


    Returns
    -------
    h : array_like, shape (D,N,M)
        The regression weights for the N features at each of the D delays for all M responses.
    b : array_like, shape (M,)
        The regression biases for all M responses.
    """

    threshold = np.float32(threshold)
    adapt_up = np.float32(adapt_up)
    adapt_down = np.float32(adapt_down)
    adapt_min = np.float32(adapt_min)
    adapt_max = np.float32(adapt_max)


    if segment_size is None:
        segment_size = np.max(delays)+1
    # index of the start of blocks to break training set into, each block is the length of max delay
    starts = np.array(range(0, X.shape[0], segment_size))
    foldstarts = [starts[x::folds] for x in range(folds)]
    foldidxs = [(f[:,None] + np.array(range(segment_size))[:, None].T).reshape(-1) for f in foldstarts]
    foldidxs = [f[f < X.shape[0]] for f in foldidxs]

    p = set(range(folds))
    val_mask = np.zeros((X.shape[0], folds*(folds-1)), dtype='float32')
    stop_mask = np.zeros((X.shape[0], folds*(folds-1)), dtype='float32')
    train_mask = np.zeros((X.shape[0], folds*(folds-1)), dtype='float32')
    for ii, i in enumerate(itertools.permutations(p, 2)):
        v = i[0]
        s = i[1]
        t = list(p - set(i))
        for tt in t:
            train_mask[foldidxs[tt], ii] = 1
        stop_mask[foldidxs[s], ii] = 1
        val_mask[foldidxs[v], ii] = 1
    
    stop_mask = np.expand_dims(stop_mask, 1)
    val_mask = np.expand_dims(val_mask, 1)
    train_mask = np.expand_dims(train_mask, 1)

    val_mask = val_mask[:,:,::(folds-1)]

    X = np.float32(X)
    y = np.float32(y[:,:,None])
    if weights is not None:
        weights = np.float32(weights.ravel()[:,None, None])

    iteration = 0
    
    # allocate weight tensors
    h0 = np.zeros((len(delays), X.shape[1], y.shape[1], (folds-1)*folds), dtype='float32')
    b0 = np.zeros((y.shape[1], (folds-1)*folds), dtype='float32')
    h_best = np.zeros((len(delays), X.shape[1], y.shape[1], (folds-1)*folds), dtype='float32')
    b_best = np.zeros((y.shape[1], (folds-1)*folds), dtype='float32')
    activeset = np.zeros((X.shape[1], y.shape[1], (folds-1)*folds), dtype=bool)
    best_err = np.float32(np.ones((y.shape[1], (folds-1)*folds))*np.inf)
    adapt = np.ones((y.shape[1], (folds-1)*folds), dtype='float32')
    badcnt = 0

    while iteration < maxiter:

        # calucate current error
        yhat = dotdelay(X, h0, b0, delays)

        (grad, err) = err_function(y, yhat, weights=weights)

        # get error on test set
        test_err = np.nansum(err * stop_mask, axis=0) / np.nansum(stop_mask, axis=0)

        # find which ones improved and update lowest errors
        better_idx = test_err < best_err
        best_err[better_idx] = test_err[better_idx]

        # update best weight vectors
        h_best[:, :, better_idx] = h0[:, :, better_idx]
        b_best[better_idx] = b0[better_idx]

        if iteration > 0:
            # backtrack if test set error does not improve
            h0[:, :, ~better_idx] -= step*adapt[~better_idx]*gw[:, :, ~better_idx]
            b0[~better_idx] -= step*gb[~better_idx]

            # replace error on back tracked models
            grad[:, ~better_idx] = grad_prev[:, ~better_idx]
            err[:, ~better_idx] = err_prev[:, ~better_idx]

            # adapt step sizes up or down
            adapt *= (adapt_up*better_idx + adapt_down*~better_idx)
            adapt = np.minimum(np.maximum(adapt, adapt_min), adapt_max)

        grad_prev = grad
        err_prev = err

        # calucate gradient
        (gw, gb) = graddelay(X, delays, grad, train_mask)

        # threshold gradient
        (gw, activeset) = threshold_grad(gw, activeset, threshold=threshold)

        # update weights
        h0 += step*adapt*gw
        b0 += step*gb

        print("iteration "+str(iteration)+" : error : "+str(np.nanmean(err)) + " : better : "+str(sum(better_idx)))

        # check if things are still improving and break if not
        if np.all(~better_idx[:]):
            badcnt += 1
        else:
            badcnt = 0

        if badcnt > 10:
            break

        iteration += 1

    allzeroidx= np.all(h_best==0, axis=(0,1))
    h_best[:,:,allzeroidx]=np.NaN
    b_best[allzeroidx]=np.NaN

    h_best_mean = np.zeros(h_best[:,:,:,0].shape + (folds,), dtype='float32')
    b_best_mean = np.zeros(b_best[:,0].shape + (folds,), dtype='float32')

    for ff in range(folds):
        idx = np.arange(folds-1) + ff*(folds-1)
        h_best_mean[:,:,:,ff] = np.nanmean(h_best[:,:,:,idx], axis=-1)
        b_best_mean[:,ff] = np.nanmean(b_best[:,idx], axis=-1)

    yhat = dotdelay(X, h_best_mean, b_best_mean, delays)

    (grad, err) = err_function(y, yhat, weights=weights)

    val_err = np.nansum(err * val_mask, axis=0) / val_mask.sum(axis=0)

    train_mask = (~val_mask.astype('bool')).astype('float32')

    train_err = np.nansum(err * train_mask, axis=0) / train_mask.sum(axis=0)

    yhat = soft_rect(yhat)

    val_corrs = np.zeros(val_err.shape)
    train_corrs = np.zeros(train_err.shape)

    if weights is None:
        weights = np.ones_like(y[:,0])

    for ff in range(folds):
        fidx = np.where(val_mask[:,:,ff])[0]
        fidx2 = np.where(train_mask[:,:,ff])[0]
        val_corrs[:,ff] = weighted_correlation(y[fidx,:,0], yhat[fidx,:,ff], weights[fidx])
        train_corrs[:,ff] = weighted_correlation(y[fidx2,:,0], yhat[fidx2,:,ff], weights[fidx2])
        # for nn in range(y.shape[1]):
        #     val_corrs[nn,ff] = np.corrcoef(y[fidx,nn,0], yhat[fidx,nn,ff])[0][1]

    return (h_best_mean, b_best_mean, val_err, val_corrs, train_err, train_corrs)


def threshold_grad_desc_bag(X, y, delays, folds=10, step=.0001, testsize=.2, maxiter=150, threshold=[.3, .4, .5, .6, .7, .8, .9],
                            adapt_up=1.2, adapt_down=.5, adapt_min=.0001, adapt_max=10000, err_function=lsq, weights=None):

    """Uses threshold gradient descent to find a linear transformation of [X] with time [delays] that approximates [y].
    This procedure is repeated for [folds] in which each parameter in [threshold] is tested on a randomly
    sampled fraction of the data specified by [testsize]. The best model weights from each fold are averaged together
    and returned as the regression weights, i.e. bagged (bootstrap aggregation).

    Parameters
    ----------
    X : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    y : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
    delays : list or array_like, shape (D,)
        The time delays to include in the model. range(5) works fine for voxels.
    folds : integer (default: 10)
        The number of times to repeat the weight estimation.
    step : float in [0..1] (default: .0001)
        The step size to use in the gradient descent.
    testsize : float in [0..1] (default: .2)
        The fraction of the data to use for early stopping and choosing the threshold in each fold
    maxiter : integer (default: 150)
        The maximum number of iterations to run the gradient descent
    threshold : list or array_like, shape (T,) (default: [.3,.4,.5,.6,.7,.8,.9])
        The values of the threshold to test.
    adapt_up : float > 1. (default: 1.2)
        The step size is adaptively made larger by multiplying by this value when error on the test set decreases
    adapt_down : float < 1. (default: .5)
        The step size is adaptively made smaller by multiplying by this value when error on the test set increases
    adapt_min : float (default: .0001)
        The floor on the range of adaptation
    adapt_max : float (default: 10000)
        The ceiling on the range of adaptation


    Returns
    -------
    h : array_like, shape (D,N,M)
        The regression weights for the N features at each of the D delays for all M responses.
    b : array_like, shape (M,)
        The regression biases for all M responses.
    """

    # index of the start of blocks to break training set into, each block is the length of max delay
    starts = np.array(range(0, X.shape[0], np.max(delays)+1))

    # allocate weight tensors
    h = np.zeros((len(delays), X.shape[1], y.shape[1], folds), dtype='float32')
    b = np.zeros((y.shape[1], folds), dtype='float32')

    if weights is not None:
        weights = weights.ravel()[:,None]

    for fold in range(folds):
        # initialize best error across thresholds to inf
        best_err = np.ones(y.shape[1], dtype='float32')*np.inf
        # sample block starts and fill out to create test index
        testidx = (np.random.permutation(starts)[:np.int(np.floor(testsize*len(starts))), None] + np.array(range(np.max(delays)+1))[:, None].T).reshape(-1)
        testidx = testidx[testidx < (X.shape[0])]

        for thresh in threshold:
            (h0, b0, best_err0) = threshold_grad_desc(X, y, delays, testidx, step, maxiter, thresh, adapt_up, adapt_down, adapt_min, adapt_max, err_function, weights=weights)
            better_idx = best_err0 < best_err
            best_err[better_idx] = best_err0[better_idx]
            h[:, :, better_idx, fold] = h0[:, :, better_idx]
            b[better_idx, fold] = b0[better_idx]

    h = np.nanmean(h, axis=-1)
    b = np.nanmean(b, axis=-1)

    return (h, b)


def threshold_grad_desc(X, y, delays, testidx, step=.0001, maxiter=150, threshold=.8, adapt_up=1.2, adapt_down=.5, adapt_min=.0001, adapt_max=10000, err_function=lsq, weights=None):
    """Uses threshold gradient descent to find a linear transformation of [X] with time [delays] that approximates [y].
    At each iteration paramters whose gradient magnitude exceeds [threshold]*(maximum gradient magnitude) are updated.
    If a given parameter exceeds threshold, all its delays are updated as well. All updated parameters are added to an
    active set which. All parameters in the active set updated at each iteration, whether or not they exceed the threshold.
    The error at each iteration is tested on a subset of the data specified by [testidx]. Step sizes are adapted up and down
    to maximize performance on this test set until no improvements can be made or the maximum number of iterations is reached.
    The best model weights according the test set are returned.


    Parameters
    ----------
    X : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    y : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
    delays : list or array_like, shape (D,)
        The time delays to include in the model. range(5) works fine for voxels.
    testidx : list or array_like, shape (H,)
        The indices to use as the early stopping set.
    step : float in [0..1] (default: .0001)
        The step size to use in the gradient descent.
    testsize : float in [0..1] (default: .2)
        The fraction of the data to use for early stopping and choosing the threshold in each fold
    maxiter : integer (default: 150)
        The maximum number of iterations to run the gradient descent
    threshold : list or array_like, shape (T,) (default: [.3,.4,.5,.6,.7,.8,.9])
        The values of the threshold to test.
    adapt_up : float > 1. (default: 1.2)
        The step size is adaptively made larger by multiplying by this value when error on the test set decreases
    adapt_down : float < 1. (default: .5)
        The step size is adaptively made smaller by multiplying by this value when error on the test set increases
    adapt_min : float (default: .0001)
        The floor on the range of adaptation
    adapt_max : float (default: 10000)
        The ceiling on the range of adaptation


    Returns
    -------
    h : array_like, shape (D,N,M)
        The regression weights for the N features at each of the D delays for all M responses.
    b : array_like, shape (M,)
        The regression biases for all M responses.
    """
    iteration = 0
    
    h0 = np.zeros((len(delays), X.shape[1], y.shape[1]), dtype='float32')
    b0 = np.zeros((y.shape[1]), dtype='float32')
    h_best = np.zeros((len(delays), X.shape[1], y.shape[1]), dtype='float32')
    b_best = np.zeros((y.shape[1]), dtype='float32')
    activeset = np.zeros((X.shape[1], y.shape[1]), dtype=bool)
    best_err = np.ones(y.shape[1], dtype='float32')*np.inf
    adapt = np.ones(y.shape[1], dtype='float32')
    badcnt = 0

    while iteration < maxiter:

        # calucate current error
        yhat = dotdelay(X, h0, b0, delays)

        (grad, err) = err_function(y, yhat, weights=weights)

        # get error on test set
        err_t = err[testidx, :]
        # remove test set error so it doesn't affect gradient calculation
        err[testidx, :] = 0
        grad[testidx, :] = 0

        # get test set error
        test_err = np.nanmean(err_t, axis=0)

        # find which ones improved and update lowest errors
        better_idx = test_err < best_err
        best_err[better_idx] = test_err[better_idx]

        # update best weight vectors
        h_best[:, :, better_idx] = h0[:, :, better_idx]
        b_best[better_idx] = b0[better_idx]

        if iteration > 0:
            # backtrack if test set error does not improve
            h0[:, :, ~better_idx] -= step*adapt[~better_idx]*gw[:, :, ~better_idx]
            b0[~better_idx] -= step*gb[~better_idx]

            # replace error on back tracked models
            grad[:, ~better_idx] = grad_prev[:, ~better_idx]
            err[:, ~better_idx] = err_prev[:, ~better_idx]

            # adapt step sizes up or down
            adapt *= (adapt_up*better_idx + adapt_down*~better_idx)
            adapt = np.minimum(np.maximum(adapt, adapt_min), adapt_max)

        grad_prev = grad
        err_prev = err

        # calucate gradient
        (gw, gb) = graddelay(X, delays, grad)

        # threshold gradient
        (gw, activeset) = threshold_grad(gw, activeset, threshold=threshold)

        # update weights
        h0 += step*adapt*gw
        b0 += step*gb

        print("iteration "+str(iteration)+" : error : "+str(np.nanmean(err)) + " : better : "+str(sum(better_idx)))

        # check if things are still improving and break if not
        if np.all(~better_idx[:]):
            badcnt += 1
        else:
            badcnt = 0

        if badcnt > 10:
            break

        iteration += 1

    allzeroidx= np.all(h_best==0, axis=(0,1))
    h_best[:,:,allzeroidx]=np.NaN
    b_best[allzeroidx]=np.NaN
    return (h_best, b_best, best_err)


def threshold_grad_desc_basis_bag(X, y, folds=10, step=.0001, testsize=.2, maxiter=150, threshold=[.3, .4, .5, .6, .7, .8, .9],
                                  adapt_up=1.2, adapt_down=.5, adapt_min=.0001, adapt_max=10000, weights=None):

    """Uses threshold gradient descent to find a linear transformation of [X] with time [delays] that approximates [y].
    This procedure is repeated for [folds] in which each parameter in [threshold] is tested on a randomly
    sampled fraction of the data specified by [testsize]. The best model weights from each fold are averaged together
    and returned as the regression weights, i.e. bagged (bootstrap aggregation).

    Parameters
    ----------
    X : array_like, shape (B, TR, N)
        Training stimuli with B basis projections, TR time points and N features. Each feature should be Z-scored across time.
    y : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
    folds : integer (default: 10)
        The number of times to repeat the weight estimation.
    step : float in [0..1] (default: .0001)
        The step size to use in the gradient descent.
    testsize : float in [0..1] (default: .2)
        The fraction of the data to use for early stopping and choosing the threshold in each fold
    maxiter : integer (default: 150)
        The maximum number of iterations to run the gradient descent
    threshold : list or array_like, shape (T,) (default: [.3,.4,.5,.6,.7,.8,.9])
        The values of the threshold to test.
    adapt_up : float > 1. (default: 1.2)
        The step size is adaptively made larger by multiplying by this value when error on the test set decreases
    adapt_down : float < 1. (default: .5)
        The step size is adaptively made smaller by multiplying by this value when error on the test set increases
    adapt_min : float (default: .0001)
        The floor on the range of adaptation
    adapt_max : float (default: 10000)
        The ceiling on the range of adaptation


    Returns
    -------
    h : array_like, shape (B,N,M)
        The regression weights for the N features at each of the B basis for all M responses.
    b : array_like, shape (M,)
        The regression biases for all M responses.
    """

    # index of the start of blocks to break training set into, each block is the length of max delay
    starts = np.array(range(0, X.shape[1], 5))

    y[np.isnan(X[0, :, 0]), :] = 0
    X[np.isnan(X)] = 0

    # allocate weight tensors
    h = np.zeros((X.shape[0], X.shape[2], y.shape[1], folds), dtype='float32')
    b = np.zeros((y.shape[1], folds), dtype='float32')

    for fold in range(folds):
        # initialize best error across thresholds to inf
        best_err = np.ones(y.shape[1], dtype='float32')*np.inf
        # sample block starts and fill out to create test index
        testidx = (np.random.permutation(starts)[:np.floor(testsize*len(starts)), None] + np.array(range(5))[:, None].T).reshape(-1)
        testidx = testidx[testidx < (X.shape[1])]

        for thresh in threshold:
            (h0, b0, best_err0) = threshold_grad_desc_basis(X, y, testidx, step, maxiter, thresh, adapt_up, adapt_down, adapt_min, adapt_max)
            better_idx = best_err0 < best_err
            best_err[better_idx] = best_err0[better_idx]
            h[:, :, better_idx, fold] = h0[:, :, better_idx]
            b[better_idx, fold] = b0[better_idx]

    h = np.nanmean(h, axis=-1)
    b = np.nanmean(b, axis=-1)

    return (h, b)


def threshold_grad_desc_basis(X, y, testidx, step=.0001, maxiter=150, threshold=.8, adapt_up=1.2, adapt_down=.5, adapt_min=.0001, adapt_max=10000, err_function=lsq):
    """Uses threshold gradient descent to find a linear transformation of [X] with time [delays] that approximates [y].
    At each iteration paramters whose gradient magnitude exceeds [threshold]*(maximum gradient magnitude) are updated.
    If a given parameter exceeds threshold, all its delays are updated as well. All updated parameters are added to an
    active set which. All parameters in the active set updated at each iteration, whether or not they exceed the threshold.
    The error at each iteration is tested on a subset of the data specified by [testidx]. Step sizes are adapted up and down
    to maximize performance on this test set until no improvements can be made or the maximum number of iterations is reached.
    The best model weights according the test set are returned.


    Parameters
    ----------
    X : array_like, shape (B, TR, N)
        Training stimuli with B basis projections, TR time points and N features. Each feature should be Z-scored across time.
    y : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
    testidx : list or array_like, shape (H,)
        The indices to use as the early stopping set.
    step : float in [0..1] (default: .0001)
        The step size to use in the gradient descent.
    testsize : float in [0..1] (default: .2)
        The fraction of the data to use for early stopping and choosing the threshold in each fold
    maxiter : integer (default: 150)
        The maximum number of iterations to run the gradient descent
    threshold : list or array_like, shape (T,) (default: [.3,.4,.5,.6,.7,.8,.9])
        The values of the threshold to test.
    adapt_up : float > 1. (default: 1.2)
        The step size is adaptively made larger by multiplying by this value when error on the test set decreases
    adapt_down : float < 1. (default: .5)
        The step size is adaptively made smaller by multiplying by this value when error on the test set increases
    adapt_min : float (default: .0001)
        The floor on the range of adaptation
    adapt_max : float (default: 10000)
        The ceiling on the range of adaptation


    Returns
    -------
    h : array_like, shape (D,N,M)
        The regression weights for the N features at each of the D delays for all M responses.
    b : array_like, shape (M,)
        The regression biases for all M responses.
    """
    iteration = 0
    h0 = np.zeros((X.shape[0], X.shape[2], y.shape[1]), dtype='float32')
    b0 = np.zeros((y.shape[1]), dtype='float32')
    h_best = np.zeros((X.shape[0], X.shape[2], y.shape[1]), dtype='float32')
    b_best = np.zeros((y.shape[1]), dtype='float32')
    activeset = np.zeros((X.shape[2], y.shape[1]), dtype=bool)
    best_err = np.ones(y.shape[1], dtype='float32')*np.inf
    adapt = np.ones(y.shape[1], dtype='float32')
    badcnt = 0

    while iteration < maxiter:

        # calucate current error
        yhat = dotbasis(X, h0, b0)

        (grad, err) = err_function(y, yhat)
        # err = y - yhat
        # get error on test set
        err_t = err[testidx, :]
        # remove test set error so it doesn't affect gradient calculation
        err[testidx, :] = 0
        # get test set error
        test_err = np.nanmean(err_t, axis=0)

        # find which ones improved and update lowest errors
        better_idx = test_err < best_err
        best_err[better_idx] = test_err[better_idx]

        # update best weight vectors
        h_best[:, :, better_idx] = h0[:, :, better_idx]
        b_best[better_idx] = b0[better_idx]


        if iteration > 0:
            # backtrack if test set error does not improve
            h0[:, :, ~better_idx] -= step*adapt[~better_idx]*gw[:, :, ~better_idx]
            b0[~better_idx] -= step*gb[~better_idx]

            # replace error on back tracked models
            grad[:, ~better_idx] = grad_prev[:, ~better_idx]
            err[:, ~better_idx] = err_prev[:, ~better_idx]

            # adapt step sizes up or down
            adapt *= (adapt_up*better_idx + adapt_down*~better_idx)
            adapt = np.minimum(np.maximum(adapt, adapt_min), adapt_max)

        grad_prev = grad
        err_prev = err

        # calucate gradient
        (gw, gb) = gradbasis(X, grad)

        # threshold gradient
        (gw, activeset) = threshold_grad(gw, activeset, threshold=threshold)

        # update weights
        h0 += step*adapt*gw
        b0 += step*gb

        print("iteration " + str(iteration)+" : error : " + str(np.nanmean(err)) + " : better : " + str(sum(better_idx)))

        # check if things are still improving and break if not
        if np.all(~better_idx[:]):
            badcnt += 1
        else:
            badcnt = 0

        if badcnt > 10:
            break

        iteration += 1

    return (h_best, b_best, best_err)


def threshold_grad(grad, activeset, threshold=.5):
    absgrad = np.max(np.abs(grad), axis=0)
    activeset = np.bitwise_or(activeset, absgrad >= threshold*np.max(absgrad, axis=0))
    grad[:, ~activeset] = 0
    return (grad, activeset)


def dotdelay(X, h, b, delays):
    outs = np.tensordot(X, h, axes=(1, 1))
    for ti, thisshift in enumerate(delays):
        outs[:, ti] = np.roll(outs[:, ti], thisshift, axis=0)
        if thisshift >= 0:
            outs[:thisshift, ti] = 0
        else:
            outs[outs.shape[0]+thisshift:, ti] = 0
    return np.nansum(outs, axis=1) + b


def graddelay(X, delays, err, mask=None):
    if mask is not None:
        err *= mask
    delout_array = np.zeros((len(delays),) + err.shape, dtype='float32')

    for ti, thisshift in enumerate(delays):
        thisshift2 = -thisshift
        delout_array[ti] = np.roll(err, thisshift2, axis=0)
        if thisshift2 >= 0:
            delout_array[ti, :thisshift2] = 0
        else:
            delout_array[ti, err.shape[0]+thisshift2:] = 0

    if mask is None:
        gw = np.transpose(np.tensordot(X.T, delout_array, (1, 1)), (1,0)+tuple(range(2, np.ndim(delout_array))))/X.shape[0]
        gb = np.nanmean(err, axis=0)
    else:
        gw = np.transpose(np.tensordot(X.T, delout_array, (1, 1)), (1,0)+tuple(range(2, np.ndim(delout_array))))/np.sum(mask, axis=0)
        gb = np.nansum(err, axis=0)/np.sum(mask, axis=0)
    return (gw, gb)


def dotbasis(X, h, b):
    return np.tensordot(X, h, axes=[(0,2), (0,1)]) + b


def gradbasis(X, err):
    gw = np.tensordot(np.transpose(X, (0, 2, 1)), err, (2, 0))/X.shape[1]
    gb = np.nanmean(err, axis=0)
    return (gw, gb)


def project_HRF_basis(X, h):
    outs = np.zeros((h.shape[1], X.shape[0], X.shape[1]), dtype='float32')
    for ti in range(h.shape[0]):
        outs += h[ti, :, None, None]*np.roll(X, ti, axis=0)
    outs[:, :h.shape[0]-1, :] = np.nan
    return outs



def weighted_correlation(x, y, w=None):
    if w is None:
        w = np.ones(len(x))
    w = w.ravel()[:,None]
    mx = np.nansum(w*x, axis=0) / np.nansum(w, axis=0)
    my = np.nansum(w*y, axis=0) / np.nansum(w, axis=0)

    sxx = np.nansum(w*(x - mx)**2, axis=0) / np.nansum(w, axis=0)
    syy = np.nansum(w*(y - my)**2, axis=0) / np.nansum(w, axis=0)
    sxy = np.nansum(w*(y - my)*(x - mx), axis=0) / np.nansum(w, axis=0)

    return sxy / np.sqrt(sxx*syy)