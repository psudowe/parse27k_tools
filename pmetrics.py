from __future__ import print_function
import numpy as np

from itertools import izip
from matplotlib import pyplot as plt

"""
Provides a number of useful evaluation metrics

And helper-routines to compute the metrics
"""

__version__ = '0.2'

#------------------------------------------------------------------------------
# metrics
def average_precision_everingham(groundtruth, predictions):
    """
    AP score as defined by Everingham et al. for the PASCAL VOC
    Reference:
    "The 2005 PASCAL Visual Object Classes Challenge"
    Lecture Notes in Computer Science Vol. 3944
    Springer 2005

    Parameters:
      groundtruth: iterable of labels (valid values: -1,0,1)
        0 labels will be ignored in the computation.
      predictions: iterable of predictions (valid values: (-inf, inf) )
        The predictions need to be converted to a continous score before,
        i.e. the commonly used softmax output cannot be used directly.

    Return:
      AP score
    """
    pr, rec = precision_recall(groundtruth, predictions)

    recall_values = np.linspace(0,1,11) # see eq. in reference
    eleven_precisions = []
    for r_thresh in recall_values:
        # find the max precision value of all threholds where recall is larger than r
        pr_values = [ p for p,r in zip(pr,rec) if r >= r_thresh ]
        if len(pr_values) == 0:
            eleven_precisions.append(0.)
        else:
            eleven_precisions.append(np.max(pr_values))

    return np.mean(eleven_precisions)

def precision_recall(groundtruth, predictions):
    """
    Computes precision-recall
    For convenience, it handles the special case of possible
    0-labels in the groundtruth, by ignoring predictions for such examples.

    Note:
      This seems very lengthy and slowish.
      However, in this formulation it is easy to read (and check) and
      easily allows to handle special cases like our NA labels in the groundtruth.

    Parameters:
      groundtruth: iterable of labels (valid values: -1,0,1)
        0 labels will be ignored in the computation.
      predictions: iterable of predictions (valid values: (-inf, inf) )
        The predictions need to be converted to a continous score before,
        i.e. the commonly used softmax output cannot be used directly.

    Return:
      tuple of (precision, recall) values
    """
    # filter NA groundtruth and sort by prediction scores
    filtered_predictions = [(p, g) for p, g in izip(predictions, groundtruth) if g != 0]
    sorted_predictions = sorted(filtered_predictions, key=lambda x: x[0])

    tp = len([x for x,g in sorted_predictions if g > 0]) # all positives
    fp = len([x for x,g in sorted_predictions if g < 0]) # all negatives
    # initially all examples are classified as positives
    # then we sweep through the data and adapt tp & fp accordingly
    tn = 0
    fn = 0
    precision = np.zeros(len(sorted_predictions)+1)
    recall    = np.zeros(len(sorted_predictions)+1)
    # handle the special case of the first threshold value
    # where everything is classified positive -> perfect recall
    precision[0] = tp / float(fp + tp)
    recall[0]    = tp / float(tp + fn)
    for idx, pg in enumerate(sorted_predictions):
        _, g = pg
        if g < 0:
            tn += 1
            fp -= 1
        elif g > 0:
            tp -= 1
            fn += 1

        if tp <= 0:
            precision[idx+1] = 0.
            recall[idx+1] =  0.
        else:
            precision[idx+1] = tp / float(fp + tp)
            recall[idx+1] =    tp / float(tp + fn)
    return precision, recall

def accuracy(confmat, ignore_na=False):
    """
    Parameters: confusion matrix (columns - predictions - rows groundtruth)
    Return: Accuracy percentage (0-100%)

    ignore_na -- if True (default False!) -- this will ignore the first class
    which by convention is the NA class
    """
    if confmat.shape[0] != confmat.shape[1]:
        raise ValueError('non-square confusion matrix')
    if len(confmat.shape) > 2:
        raise ValueError('more than 2 dimensions in confusion matrix')
    N = confmat.shape[0]
    correct = 0
    errors = 0
    for pr_idx in range(N):
        for gt_idx in range(N):
            if ignore_na:
                if pr_idx == 0 or gt_idx == 0:
                    continue
            if pr_idx == gt_idx:
                correct += confmat[pr_idx, gt_idx]
            else:
                errors += confmat[pr_idx, gt_idx]
    accuracy = 100. * float(correct) / (correct + errors)
    return accuracy

def balanced_error_rate(confmat):
    """
    Compute balanced error rate

    Parameters: confmat as computed by compute_confmat.
    Returns: float BER
    """
    if confmat.shape[0] != confmat.shape[1]:
        raise ValueError('non-square confusion matrix')
    if len(confmat.shape) > 2:
        raise ValueError('more than 2 dimensions in confusion matrix')
    N = confmat.shape[0]
    per_class_err = np.zeros(N)
    for idx in range(N):
        per_class_err[idx] = (np.sum(confmat[idx,:]) - confmat[idx,idx]) / np.sum(confmat[idx,:])
    return np.mean(per_class_err)

#------------------------------------------------------------------------------
# helper routines
def softmax_to_confmat(gtlabels, predictions):
    """
    gtlabel - index of groundtruth
    predictions - label

    returns confusion matrix
    """
    N = np.max(gtlabels)
    confmat = np.zeros((N+1,N+1))
    for gt, pred in zip(gtlabels, predictions):
        pred_idx = np.argmax(pred)
        confmat[gt][pred_idx] += 1
    return confmat

def compute_confmat(gtlabels, predictions):
    """ compute a confusion matrix
    Parameters:
      - gtlabels -- the groundtruth (as index in range [0..N) for N classes
      - predictions -- same format as groundtruth
    Returns:
      confusion matrix
    """
    N = np.max(gtlabels)
    NP = np.max(predictions)
    if NP > N:
        print('WARNING: predictions and groundtruth do not seem to match')
        N = NP
    confmat = np.zeros((N+1,N+1))
    for gt, pred in zip(gtlabels, predictions):
        confmat[gt][pred] += 1
    return confmat

def softmax_prediction_to_binary(predictions, ignore_na=True):
    """
    Transform a softmax prediction for a binary (2+1) attribute, to
    a single value, while ignoring the NA class (first column)
    This transformation obviously only makes sense for (2 or 2+1 columns).

    Parameters:
      predictions     np.array  N x K entries (K=3)
          this assumes that K=0 is the NA label

    Return:
      output          N x 1 array with class scores
    """
    if predictions.shape[1] > 3:
        raise ValueError('can only convert binary attributes (binary + N/A)!')
    if not ignore_na and np.any( predictions[:,0] > 0 ):
        raise ValueError('predictions for NA class - but ignore_na=False')
    return predictions[:,1]

def labels_to_binary(labels):
    """
        computes translation:
            (0 == 0, 1 == 1, 2 == -1)
    Take
        labels  N x 3 entries
    Return
        labels  N entries (-1,0,1)
    """
    if np.any( labels > 2 ):
        raise ValueError('too large gt value observed')
    if np.any( labels < 0):
        raise ValueError('label already has a negative value')
    out = labels
    out[ labels == 2 ] = -1
    return out
