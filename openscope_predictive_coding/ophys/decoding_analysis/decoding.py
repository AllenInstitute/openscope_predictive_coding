#Misc
import pdb,glob,fnmatch, sys
import os, time, datetime
import glob, fnmatch

#Base
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
import multiprocessing as mp

#Plot
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib import pyplot as plt

#Plotting Params
sns.set_style("ticks")
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.major.pad']='10'

color_names=['windows blue','red','amber','faded green','dusty purple','orange','steel blue','pink',
             'greyish','mint','clay','light cyan','forest green','pastel purple','salmon','dark brown',
             'lavender','pale green','dark red','gold','dark teal','rust','fuchsia','pale orange','cobalt blue']
color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)

#Decoding
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

#Decoding Params
nProcesses = 25
nShuffles = 100

#Predictive Coding Github Repository
sys.path.append('/home/dwyrick/Git/openscope_predictive_coding/')
from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
cache_dir = '/srv/data/AllenInst/opc_analysis'

mainseq_ids = [68, 78, 13, 26]
oddball_ids = [6, 17, 22, 51, 71, 89, 103, 110, 111, 112]
transctrl_dict = {68: (68,78), 78: (78,13), 13: (13,26), 26: (26,68)}

# variable: 'summed_events' or 'mean_response' when we're using deconvolved events
#           'max_response' or 'mean_response' when we're using raw dfof
def create_psuedopopulation(manifest, area, block='None', use_events=True, variable='summed_events'):
    
    #Get experimental IDs for area of interest
    experiment_ids = manifest[(manifest.imaging_area==area)].experiment_id.values

    X_list = []
    for eID in experiment_ids:
        ## Temporary Fix: For some reason this experimental ID throws errors for me -D. Wyrick
        if int(eID) == 848691390:
            continue
            
        ## Load in dataset
        dataset = OpenScopePredictiveCodingDataset(int(eID), cache_dir)

        #Create analysis object
        analysis = ResponseAnalysis(dataset, use_events=use_events) 

        # Get stimulus table for all of the presentations for each of the stimulus types
        stimulus_df = dataset.stimulus_table

        #Get data from this block
        stim_ids = stimulus_df[stimulus_df.session_block_name == block].index.values

        #Get response x-array and take deconvolved events
        xresp = analysis.get_response_xr(block)

        #Append this experiments data
        X_list.append(xresp[variable].loc[stim_ids])
        
    #Concatenate neurons from all experiments to create pseudopopulation vectors
    X_responses = xr.concat(X_list,'cell_specimen_id')
    
    print('{} pseudopopulation created for the {} block: {} neurons from {} experiments'.format(area,block, X_responses.shape[-1],len(X_list)))
    return X_responses, stimulus_df

# trial_type options:
#    'ABCD'  -> Just main sequence (MS) images
#     X'     -> Just oddball (X) images
#    'ABCDX' -> MS images and the  oddball images immediately following them; to be more precise, it's DABCX or ABCD---X depending on seq_dist
#    'XABCD' -> "  " proceeding them
#
# seq_dist: for ABCDX and XABCD trial types, how many sequences of ABCD apart should the MS and oddball trials be?
# e.g. ABCD----X where the - are trials we skip. this would be seq_dist = 1, whereas DABCX would be seq_dist = 0
# e.g. XABCD is seq_dist = 1
#
# trial_indy: for trial type ABCD, this parameter indicates which 240 trials you want to use (60 trials per stimulus)
#Get stimulus presentation IDs for images in the main sequence in the oddball blocks
#
# TODO: this is only set up for the oddball block and randomized_ctrl blocks really
        
def match_trials(X_responses, stimulus_df, block='oddball', trial_type='ABCD', seq_dist=1, trial_indy=slice(None)):

    #Get the starting stimulus presentation ID for this block
    starting_stimID = stimulus_df[(stimulus_df.session_block_name == block)].index.values[0]
    
    #Get trial indices where MS images and oddball images were presented
    MS_stimIDs = stimulus_df[(stimulus_df.session_block_name == block) & (stimulus_df.image_id.isin(mainseq_ids))].index.values
    OB_stimIDs = stimulus_df[(stimulus_df.session_block_name == block) & (stimulus_df.image_id.isin(oddball_ids))].index.values
    
    if block == 'oddball':
        #If we're in the oddball block, make sure we only take MS images that are at least 2 sequences away from an oddball
        if trial_type == 'ABCD':        
            seq_ids_wo_oddballs = []
            for sID in MS_stimIDs:
                #If it's the first 2 sequencces, add them to the list; no oddballs have been presented
                if (sID-12) < starting_stimID:
                    seq_ids_wo_oddballs.append(sID)
                    continue
                #If the current sequence, and the last sequence, and 2 sequences ago all ended without an oddball, use that trial; look at stimulus_key 
                if all([stimulus_df.loc[sID-4*ii]['stimulus_key'][-1] == mainseq_ids[-1] for ii in range(4)]):
                    seq_ids_wo_oddballs.append(sID)
            MS_stimIDs = np.array(seq_ids_wo_oddballs)
            
            #Use presentation IDs to select data we're interested in
            X_subset = X_responses.loc[MS_stimIDs[trial_indy]]
            Y_subset = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values
            Y_sort = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values

        elif trial_type == 'ABCDX':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            if seq_dist > 0:
                #We're taking ABCD---X
                prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - 4*seq_dist - ii for ii in range(4)]))
            elif seq_dist == 0:
                #We're taking DABCX, where D is shared above if seq_dist == 1
                prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((prevMS_stimIDs,OB_stimIDs))
            X_subset = X_responses.loc[stim_ids]

            #These are the class labels we will use to construct a classifier; note, I've labeled all of the oddballs 1
            Y_subset =np.concatenate((stimulus_df.loc[prevMS_stimIDs]['image_id'],np.repeat(1,100)))
            #But to ensure equal proportions of each oddball in a particulat fold of the cross-validation, we create Y_sort
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values

        elif trial_type == 'XABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs that occur after an oddball has been presented
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + 4*seq_dist + ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((nextMS_stimIDs,OB_stimIDs))
            X_subset = X_responses.loc[stim_ids]
            Y_subset =np.concatenate((stimulus_df.loc[nextMS_stimIDs]['image_id'].values,np.repeat(1,100)))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'X':
            #Just get the oddball trials
            X_subset = X_responses.loc[OB_stimIDs]
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
            
    elif block == 'transition_control':
        #Make sure we take trials of the same transition type as in the oddball context
        #i.e. trials where image B was presented will be proceeded by tials where image A was shown
        if trial_type == 'ABCD': 
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)

            #Use presentation IDs to select data we're interested in
            X_subset = X_responses.loc[MS_stimIDs[trial_indy]]
            Y_subset = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values
            Y_sort = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values
        
        #Similarly, now we're going to include all of the DX transitions
        #In the transition context, ABCDX is almost the same as XABCD, except for which 'A' trials we want to include
        #ABCDX: (D,A),(A,B),(B,C),(C,D),(C,X)
        #XABCD: (X,A),(A,B),(B,C),(C,D),(C,X)
        elif trial_type == 'ABCDX':
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))
            X_subset = X_responses.loc[stim_ids]
            Y_subset = np.concatenate((stimulus_df.loc[MS_stimIDs]['image_id'],np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        #Similarly, now we're going to include all of the XA instead of DA transitions
        elif trial_type == 'XABCD':
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'][0] in oddball_ids):
                    seq_ids_correct_transitions.append(sID)
                #Get AB, BC, CD trials
                elif (imgID in mainseq_ids[1:]) & (transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']):
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))
            X_subset = X_responses.loc[stim_ids]
            Y_subset = np.concatenate((stimulus_df.loc[MS_stimIDs]['image_id'],np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        #Let's see if these 'A' trials are different
        elif trial_type == 'DAXA':
            #Get DA
            DA_stim_ids = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']):
                    DA_stim_ids.append(sID)
            DA_stim_ids = np.array(DA_stim_ids)
            
            #Get XA
            XA_stim_ids = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'][0] in oddball_ids):
                    XA_stim_ids.append(sID)
            XA_stim_ids = np.array(XA_stim_ids)
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((DA_stim_ids,XA_stim_ids))
            X_subset = X_responses.loc[stim_ids]
            Y_subset = np.concatenate((np.repeat(0,len(DA_stim_ids)),np.repeat(1,len(XA_stim_ids))))
            Y_sort = np.array([stimulus_df.loc[sID]['stimulus_key'][0] for sID in stim_ids])
            
        elif trial_type == 'X':
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            
            #Just get the oddball trials
            X_subset = X_responses.loc[OB_stimIDs]
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
            
    else:
        if trial_type == 'ABCD':
            #Just get the MS trials
            X_subset = X_responses.loc[MS_stimIDs]
            Y_subset = stimulus_df.loc[MS_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[MS_stimIDs]['image_id'].values
            
        elif trial_type == 'X':
            #Just get the oddball trials
            X_subset = X_responses.loc[OB_stimIDs]
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
    
    return X_subset, Y_subset, Y_sort
    

##====================##
##===== Decoding =====##

def decode_labels(X,Y,train_index,test_index,classifier='LDA',clabels=None,X_test=None,Y_test=None,shuffle=True,classifier_kws=None):
    
    #Copy training index for shuffle decoding
    train_index_sh = train_index.copy()
    np.random.shuffle(train_index_sh)
    
    #Split data into training and test sets
    if X_test is None:
        #Training and test set are from the same time interval
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        
        #Get class labels
        Y_train = Y[train_index]
        Y_test = Y[test_index]
    
    else:
        #Training and test set are from different epochs
        X_train = X
        Y_train = Y
     
    #How many classes are we trying to classify?
    class_labels,nTrials_class = np.unique(Y,return_counts=True)
    nClasses = len(class_labels)

    #Initialize Classifier
    if classifier == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif classifier == 'SVM':
        clf = svm.LinearSVC(max_iter=1E6)

    #Luca's decoder 
    if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
        nTrials_test, nNeurons = X_test.shape
        PSTH_train = np.zeros((nClasses,nNeurons))
        #Calculate PSTH templates from training data
        for iStim, cID in enumerate(class_labels):
            pos = np.where(Y_train == cID)[0]
            PSTH_train[iStim] = np.mean(X_train[pos],axis=0)
        
        Y_hat = np.zeros((nTrials_test,),dtype=int)
        for iTrial in range(nTrials_test):
            if classifier == 'Euclidean_Dist':
                #Predict test data by taking the minimum euclidean distance
                dist = [np.sum((X_test[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmin(dist)]
            else:
                #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                Rs = [np.corrcoef(PSTH_train[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                
            #Just to check whether the classifier correctly decodes the training data
#         nTrials_train = X_train.shape[0]
#         Y_hat_train = np.zeros((nTrials_train,),dtype=int)
#         for iTrial in range(nTrials_train):
#             if classifier == 'Euclidean_Dist':
#                 #Predict test data by taking the minimum euclidean distance
#                 dist = [np.sum((X_train[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
#                 Y_hat_train[iTrial] =  class_labels[np.argmin(dist)]
#             else:
#                 #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
#                 Rs = [np.corrcoef(PSTH_train[iStim],X_train[iTrial])[0,1] for iStim in range(nClasses)]
#                 Y_hat_train[iTrial] =  class_labels[np.argmax(Rs)]
#         pdb.set_trace()
    #All other classifiers
    else:
        #Fit model to the training data
        clf.fit(X_train, Y_train)

        #Predict test data
        Y_hat = clf.predict(X_test)
        Y_hat_train = clf.predict(X_train)
    
    #Calculate confusion matrix
    kfold_hits = confusion_matrix(Y_test,Y_hat,labels=clabels)
#     kfold_hits_train = confusion_matrix(Y_train,Y_hat_train,labels=clabels)
#     print(kfold_hits)
#     coef = clf.coef_
#     pdb.set_trace()

    ##===== Perform Shuffle decoding =====##
    if shuffle:
        kfold_shf = np.zeros((nShuffles,nClasses,nClasses))
        #Classify with shuffled dataset
        for iS in range(nShuffles):
            #Shuffle training indices
            np.random.shuffle(train_index_sh)
            Y_train_sh = Y[train_index_sh]

            #Initialize Classifier
            if classifier == 'LDA':
                clf_shf = LinearDiscriminantAnalysis()
            elif classifier == 'SVM':
                clf_shf = svm.LinearSVC(max_iter=1E6) #C=classifier_kws['C']
            
            #"Luca's" decoder 
            if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
                nTrials_test, nNeurons = X_test.shape
                PSTH_sh = np.zeros((nClasses,nNeurons))
                
                #Calculate PSTH templates from training data
                for iStim, cID in enumerate(class_labels):
                    pos = np.where(Y_train_sh == cID)[0]
                    PSTH_sh[iStim] = np.mean(X_train[pos],axis=0)

                Y_hat = np.zeros((nTrials_test,),dtype=int)
                for iTrial in range(nTrials_test):
                    if classifier == 'Euclidean_Dist':
                        #Predict test data by taking the minimum euclidean distance
                        dist = [np.sum((X_test[iTrial] - PSTH_sh[iStim])**2) for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmin(dist)]
                    else:
                        #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                        Rs = [np.corrcoef(PSTH_sh[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                    
            #All other classifiers
            else:
                #Fit model to the training data
                clf_shf.fit(X_train, Y_train_sh)

                #Predict test data
                Y_hat = clf_shf.predict(X_test)

            #Calculate confusion matrix
            kfold_shf[iS] = confusion_matrix(Y_test,Y_hat,labels=clabels)
    else:
        kfold_shf = np.zeros((nClasses,nClasses))

        #Return decoding results
    return kfold_hits, kfold_shf

def calculate_accuracy(results,method='L1O',plot_shuffle=False,pdfdoc=None):
    
    nClasses = results[0][0].shape[0]
    #Save results to these
    confusion_mat = np.zeros((nClasses,nClasses))
    confusion_shf = np.zeros((nClasses,nClasses))
    confusion_z = np.zeros((nClasses,nClasses))
    
    if method == 'L1O':    
        c_shf = np.zeros((nShuffles,nClasses,nClasses))
        for iK,rTuple in enumerate(results):
            loo_hits = rTuple[0] #size [nClasses x nClasses]
            loo_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]

            #Add hits to confusion matrix
            confusion_mat += loo_hits

            #Loop through shuffles
            for iS in range(nShuffles):
                #Add hits to confusion matrix
                c_shf[iS] += loo_shf[iS]

        #Calculate decoding accuracy for this leave-1-out x-validation
        confusion_mat = confusion_mat/np.sum(confusion_mat,axis=1).reshape(-1,1)

        #Loop through shuffles
        for iS in range(nShuffles):
            #Calculate shuffled decoding accuracy for this leave-1-out shuffle
            c_shf[iS] = c_shf[iS]/np.sum(c_shf[iS],axis=1).reshape(-1,1)

        #Calculate z-score 
        m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
        confusion_shf = m_shf
        confusion_z = (confusion_mat - m_shf)/s_shf

        #Get signficance of decoding 
        pvalues_loo = st.norm.sf(confusion_z)
        
        if plot_shuffle:
            #Plot shuffle distributions
            title = 'Leave-1-Out Cross-Validation'
            plot_decoding_shuffle(confusion_mat, c_shf, pvalues_loo, title,pdfdoc)
            
    elif method == 'kfold':       
        kfold_accuracies = []
        shf_accuracies = []
        kfold_zscores = []
        
        #Calculate decoding accuracy per kfold
        for iK,rTuple in enumerate(results):
            kfold_hits = rTuple[0] #size [nClasses x nClasses]
            kfold_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]

            #Normalize confusion matrix
            kfold_accuracies.append(kfold_hits/np.sum(kfold_hits,axis=1).reshape(-1,1))

            #Loop through shuffles and normalize
            c_shf = np.zeros((nShuffles,nClasses,nClasses))
            for iS in range(nShuffles):
                c_shf[iS] = kfold_shf[iS]/np.sum(kfold_shf[iS],axis=1).reshape(-1,1)

            #Calculate z-score for this kfold
            m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
            shf_accuracies.append(m_shf)
            kfold_zscores.append((kfold_accuracies[iK] - m_shf)/s_shf)

            #Get signficance of decoding 
            pvalues_kfold = st.norm.sf(kfold_zscores[iK])
            
            if plot_shuffle:
                #Plot shuffle distributions
                title = 'Shuffle Distributions for kfold {}'.format(iK)
                plot_decoding_shuffle(kfold_accuracies[iK], c_shf, pvalues_kfold, title,pdfdoc)
        
        #Take average over kfolds
        confusion_mat = np.mean(kfold_accuracies,axis=0)
        confusion_shf = np.mean(shf_accuracies,axis=0)
        confusion_z = np.mean(kfold_zscores,axis=0)
        
    return confusion_mat, confusion_shf, confusion_z
    
def cross_validate(X,Y,Y_sort=None,method='L1O',nKfold=10,classifier='LDA',classifier_kws=None,clabels=None,shuffle=True,plot_shuffle=False,parallel=False,pdfdoc=None):
    ##===== Description =====##
    #The main difference between these 2 methods of cross-validation are that kfold approximates the decoding accuracy per kfold 
    #and then averages across folds to get an overall decoding accuracy. This is faster and a better approximation of the actual
    #decoding accuracy if you have enough data. By contrast, Leave-1-out creates just 1 overall decoding accuracy by creating 
    #a classifier for each subset of data - 1, and adding those results to a final confusion matrix to get an estimate of the 
    #decoding accuracy. While this is what you have to do in low-data situations, the classifiers are very similar and thus 
    #share a lot of variance. Regardless, I've written both ways of calculating the decoding accuracy below. 
    
    if Y_sort is None:
        Y_sort = Y
        
    #Leave-1(per group)-out
    if method == 'L1O':
        _,nTrials_class = np.unique(Y,return_counts=True)
        k_fold = StratifiedKFold(n_splits=nTrials_class[0])
    #Or k-fold
    elif method == 'kfold':
        k_fold = StratifiedKFold(n_splits=nKfold)
            
    #Multi-processing module is weird and might hang, especially with jupyter; try without first
    if parallel:
        pool = mp.Pool(processes=nProcesses)
        processes = []
    results = []
    
    ##===== Loop over cross-validation =====##
    for iK, (train_index, test_index) in enumerate(k_fold.split(X,Y_sort)):
#         print(np.unique(Y[train_index],return_counts=True))
#         print(np.unique(Y_sort[train_index],return_counts=True))
#         pdb.set_trace()
        if parallel:
            processes.append(pool.apply_async(decode_labels,args=(X,Y,train_index,test_index,classifier,clabels,X_test,shuffle,classifier_kws)))
        else:
            tmp = decode_labels(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,classifier_kws)
            results.append(tmp)

    #Extract results from parallel kfold processing
    if parallel:
        results = [p.get() for p in processes]
        pool.close()
        
    ##===== Calculate decoding accuracy =====##
    confusion_mat, confusion_shf, confusion_z = calculate_accuracy(results,method,plot_shuffle,pdfdoc)
    
    return confusion_mat, confusion_shf, confusion_z


##==============================##
##===== Plotting Functions =====##

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    

def plot_decoding_shuffle(decoding_accuracy, shuffles, pvalues,title=None,pdfdoc=None):
    
    nClasses = decoding_accuracy.shape[-1]
    ## Plot shuffle distributions ##
    fig,axes = plt.subplots(1,nClasses,figsize=(18,6))
    plt.suptitle(title,y=1.01)

    #Plot the shuffle distribution with the mean decoding performance for that class
    for i in range(nClasses):
        ax = axes[i]
        sns.distplot(shuffles[:,i,i],color=cc[i],ax=ax)
        if pvalues[i,i] < 0.01:
            ax.set_title('element [{},{}], pval: {:.1e}'.format(i,i,pvalues[i,i]))
        else:
            ax.set_title('element [{},{}], pval: {:.2f}'.format(i,i,pvalues[i,i]))

        ax.vlines(decoding_accuracy[i,i], *ax.get_ylim(),LineWidth=2.5,label='Data: {:.2f}'.format(decoding_accuracy[i,i]))
        ax.vlines(np.mean(shuffles,axis=0)[i,i], *ax.get_ylim(),LineWidth=2.5,LineStyle = '--',label='Shuffle: {:.2f}'.format(np.mean(shuffles,axis=0)[i,i]))
        ax.set_xlim(xmin=0)
        ax.legend()

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)

def plot_confusion_matrices(confusion_mat, confusion_z, plot_titles=None,class_labels=None,title=None, pdfdoc=None):

    nPlots = confusion_mat.shape[0]
    nClasses = confusion_mat.shape[-1]
    
    ##===== Plotting =====##
    fig,axes = plt.subplots(1,5,figsize=(16.25,4),gridspec_kw={'wspace': 0.25,'width_ratios':(4,4,4,4,0.25)})#,
    plt.suptitle(title,y=0.95)

    for iPlot in range(nPlots):
        ax = axes[iPlot]
        pvalues = st.norm.sf(confusion_z[iPlot])
        
        #Plot actual decoding performance
#         if iPlot == 4 :
        sns.heatmap(confusion_mat[iPlot],annot=True,fmt='.2f',annot_kws={'fontsize': 16},cbar=True,cbar_ax=axes[-1],square=True,cbar_kws={'shrink': 0.25,'aspect':1},vmin=np.percentile(confusion_mat[:],5),vmax=np.percentile(confusion_mat[:],95),ax=ax)
        ax.set_title('Image {}'.format(plot_titles[iPlot]),fontsize=14)
    
        for i in range(nClasses):
            for j in range(nClasses):
                if (pvalues[i,j] < 0.05):
                    ax.text(j+0.75,i+0.25,'*',color='g',fontsize=20,fontweight='bold')
        #Labels
        if class_labels is not None:
            if iPlot == 0:
                ax.set_ylabel('Actual Stimulus Block',fontsize=12)
                ax.set_yticklabels(class_labels, rotation=90,va="center",fontsize=12)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Decoded Stimulus Block',fontsize=12)
            ax.set_xticklabels(class_labels,va="center",fontsize=12) 
            
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)
    
def plot_decoding_accuracy(confusion_mat,confusion_z,ax=None,block='randomized_ctrl',class_labels=None,title=None,annot=True,clims=None):
    #Plot decoding performance
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))#,
        
    if title is not None:
        ax.set_title(title,fontsize=14)

    pvalues = st.norm.sf(confusion_z)
    
    if clims is None:
        clims = [np.percentile(confusion_mat,1) % 0.05, np.percentile(confusion_mat,99) - np.percentile(confusion_mat,99) % 0.05]
    #Plot actual decoding performance
    sns.heatmap(confusion_mat,annot=annot,fmt='.2f',annot_kws={'fontsize': 16},cbar=True,square=True,vmin=clims[0],vmax=clims[1],cbar_kws={'shrink': 0.5},ax=ax)

    nClasses = confusion_mat.shape[0] 
    for i in range(nClasses):
        for j in range(nClasses):
            if (pvalues[i,j] < 0.05):
                ax.text(j+0.65,i+0.35,'*',color='g',fontsize=20,fontweight='bold')
    #Labels
    ax.set_ylabel('Actual Image',fontsize=14)
    ax.set_xlabel('Decoded Image',fontsize=14)
    if class_labels is not None:
        ax.set_yticklabels(class_labels,va="center",fontsize=14)
        ax.set_xticklabels(class_labels,va="center",fontsize=14) 