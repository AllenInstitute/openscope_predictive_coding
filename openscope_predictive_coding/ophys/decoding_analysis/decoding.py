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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Decoding Params
nProcesses = 25
nShuffles = 50

#Predictive Coding Github Repository
sys.path.append('/home/dwyrick/Git/openscope_predictive_coding/')
from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
cache_dir = '/srv/data/AllenInst/opc_analysis'

mainseq_ids = [68, 78, 13, 26]
oddball_ids = [6, 17, 22, 51, 71, 89, 103, 110, 111, 112]
transctrl_dict = {68: (26,68), 78: (68,78), 13: (78,13), 26: (13,26), 6: (13,6), 17: (13,17), 22: (13,22), 51: (13,51), 71: (13,71), 89: (13,89), 103: (13,103), 110: (13,110), 111: (13,111), 112: (13,112)}

from multiprocessing import Pool

##==================================================
def get_responses_per_session(manifest, area, use_events=True, variable='summed_response'):

    #Get experimental IDs for area of interest
    experiment_ids = manifest[(manifest.imaging_area==area)].experiment_id.values

    stimulus_blocks = ['randomized_control_pre','oddball','transition_control','randomized_control_post']
    X_responses = {b: {} for b in stimulus_blocks}
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

        ##========================
        for context in stimulus_blocks:
            #Get stimulus IDs
            stim_ids = stimulus_df[stimulus_df.session_block_name == context].index.values

            #Get response x-array and take deconvolved events
            xresp = analysis.get_response_xr(context)
            x_sub = xresp[variable].sel(stimulus_presentations_id = stim_ids)
            X_responses[context][eID] = x_sub
            
    return X_responses, stimulus_df

##==================================================
def _get_response_dfs_per_session(eID,use_events=True, variables=['trace','summed_response']):
    ## Load in dataset
    dataset = OpenScopePredictiveCodingDataset(int(eID), cache_dir)

    #Create analysis object
    analysis = ResponseAnalysis(dataset, use_events=use_events, regenerate_dfs = True)#, regenerate_dfs = True)
#             analysis.overwrite_analysis_files = True
            # analysis.regenerate_dfs = True

    # Get stimulus table for all of the presentations for each of the stimulus types
    stimulus_df = dataset.stimulus_table

    ##========================
    #Get randomized control pre and post       
    #Get response dataframe and take deconvolved events
    x_df = analysis.get_response_df('randomized_control_pre')
    randctrl_pre_df = x_df[['stimulus_presentations_id','cell_specimen_id','session_block_name','image_id','oddball',*variables]]

    #Get response dataframe and take deconvolved events
    x_df = analysis.get_response_df('randomized_control_post')
    randctrl_post_df = x_df[['stimulus_presentations_id','cell_specimen_id','session_block_name','image_id','oddball',*variables]]

    ##========================
    #Get oddball block
    #Get response dataframe and take deconvolved events
    x_df = analysis.get_response_df('oddball')
    oddball_df = x_df[['stimulus_presentations_id','cell_specimen_id','session_block_name','image_id','oddball',*variables]]

    ##========================
    #Get transition block
    x_df = analysis.get_response_df('transition_control')
    transctrl_df = x_df[['stimulus_presentations_id','cell_specimen_id','session_block_name','image_id','oddball',*variables]]

    #Append this experiments data
    tmp_df = pd.concat((randctrl_pre_df,oddball_df,transctrl_df,randctrl_post_df))
    return (tmp_df, stimulus_df)

##==================================================
def get_response_dfs_per_session(manifest,  use_events=True, variables=['trace','summed_response']):
    areanames = ['RSP', 'VISp', 'VISpm']
    blocks = ['randomized_control_pre','randomized_control_post','transition_control','oddball']
    
    response_df = {'VISp': {}, 'VISpm': {}, 'RSP': {}}

    # area = 'RSP'
    for area in areanames:
        #Get experimental IDs for area of interest
        experiment_ids = manifest[(manifest.imaging_area==area)].experiment_id.values
        print('\n{}: '.format(area))
        processes = []
        with Pool(20) as p: 
            for eID in experiment_ids:
                ## Temporary Fix: For some reason this experimental ID throws errors for me -D. Wyrick
                if int(eID) == 848691390:
                    continue

                processes.append(p.apply_async(_get_response_dfs_per_session,args=(eID,use_events,variables)))

            for eID, out in zip(experiment_ids,processes):
                print('{}, '.format(int(eID)),end='')
                stimulus_df= out.get()[1]
                response_df[area][eID] = out.get()[0]

    return response_df, stimulus_df

##==================================================
# variable: 'summed_events' or 'mean_response' when we're using deconvolved events
#           'max_response' or 'mean_response' when we're using raw dfof
def create_psuedopopulation(manifest, area, block='None', use_events=True, variable='summed_response'):
           
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
        X_list.append(xresp[variable].sel(stimulus_presentations_id= stim_ids))

    X_responses = xr.concat(X_list,'cell_specimen_id')
    print('{} pseudopopulation created for the {} block: {} neurons from {} experiments'.format(area,block, X_responses.shape[-1],len(X_list)))
    return X_responses, stimulus_df

##==================================================
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
            stimIDs = MS_stimIDs[trial_indy]
            #Use presentation IDs to select data we're interested in
            X_subset = X_responses.sel(stimulus_presentations_id = MS_stimIDs[trial_indy])
#             X_subset = X_responses.loc[MS_stimIDs[trial_indy]]
            Y_subset = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values
            Y_sort = stimulus_df.loc[MS_stimIDs[trial_indy]]['image_id'].values
                    
        elif trial_type == 'ABCX':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,4)])) #just get the ABC before X
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((prevMS_stimIDs,OB_stimIDs))
            # stim_ids = np.sort(np.concatenate((prevMS_stimIDs,OB_stimIDs)))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)

            Y_subset =np.concatenate((stimulus_df.loc[prevMS_stimIDs]['image_id'],np.repeat(1,100)))
            # Y_subset = stimulus_df.loc[stim_ids]['image_id'].values
            
            #But to ensure equal proportions of each oddball in a particulat fold of the cross-validation, we create Y_sort
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'X-ABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,4)])) #just get the ABC before X
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.sort(np.concatenate((OB_stimIDs,nextMS_stimIDs)))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)

            Y_subset = stimulus_df.loc[stim_ids]['image_id'].values
            #But to ensure equal proportions of each oddball in a particulat fold of the cross-validation, we create Y_sort
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
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
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)

            #These are the class labels we will use to construct a classifier; note, I've labeled all of the oddballs 1
            Y_subset =np.concatenate((stimulus_df.loc[prevMS_stimIDs]['image_id'],np.repeat(1,100)))
            # Y_subset = stimulus_df.loc[stim_ids]['image_id'].values
            #But to ensure equal proportions of each oddball in a particulat fold of the cross-validation, we create Y_sort
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values

        elif trial_type == 'XABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs that occur after an oddball has been presented
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + 4*seq_dist + ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((nextMS_stimIDs,OB_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset =np.concatenate((stimulus_df.loc[nextMS_stimIDs]['image_id'].values,np.repeat(1,100)))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'X':
            #Just get the oddball trials
            X_subset = X_responses.sel(stimulus_presentations_id = OB_stimIDs)
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
        elif trial_type == 'DAXA':
            #Get DA
            #Use the oddball stimulus presentation IDs to identify the main sequence image A that occurs after D but before X; DABCX
            DA_stimIDs = sorted(OB_stimIDs - 3 - 4*seq_dist)
            
            #And the A that occurs after X
            XA_stimIDs = sorted(OB_stimIDs + 1)
              
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((DA_stimIDs,XA_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((np.repeat(0,len(DA_stimIDs)),np.repeat(1,len(XA_stimIDs))))
            Y_sort = np.array([stimulus_df.loc[sID - 1]['image_id'] for sID in stim_ids])
            
        elif trial_type == 'DX':
            #Use the oddball stimulus presentation IDs to identify the main sequence image D before X; DABCX
            D_stimIDs = sorted(OB_stimIDs - 4*seq_dist)
            
            stim_ids = np.concatenate((D_stimIDs,OB_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((np.repeat(0,len(D_stimIDs)),np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'XD':
            #Use the oddball stimulus presentation IDs to identify the main sequence image D after  X; XABCD
            D_stimIDs = sorted(OB_stimIDs + 4*seq_dist)
            
            stim_ids = np.concatenate((D_stimIDs,OB_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((np.repeat(0,len(D_stimIDs)),np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
 
        elif trial_type == 'all':
            #Just get the oddball trials
            X_subset = X_responses
            stimIDs = X_subset.coords['stimulus_presentations_id'].values
            Y_subset = stimulus_df.loc[stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[stimIDs]['image_id'].values
            
            
    elif block == 'transition_control':
        #Make sure we take trials of the same transition type as in the oddball context
        #i.e. trials where image B was presented will be proceeded by tials where image A was shown
        if trial_type == 'ABCD': 
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            stimIDs = MS_stimIDs
            #Use presentation IDs to select data we're interested in
            X_subset = X_responses.sel(stimulus_presentations_id = MS_stimIDs[trial_indy])
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
                if stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
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
                elif (imgID in mainseq_ids[1:]) & (stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]):
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
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((stimulus_df.loc[MS_stimIDs]['image_id'],np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['stimulus_key'].values
    
        #Similarly, now we're going to include all of the XA instead of DA transitions
        elif trial_type == 'CX':

            #Get CX trials
            stimIDs_correct_transitions = []
            for iID in oddball_ids:
                #CX trials
                stimIDs =  stimulus_df[(stimulus_df.session_block_name == 'transition_control') & (stimulus_df.image_id == iID)  & (stimulus_df.stimulus_key == transctrl_dict[iID])].index.values
                stimIDs_correct_transitions.append(stimIDs-1) #Append C trials
                stimIDs_correct_transitions.append(stimIDs) #Append X trials
                
            #Use presentation IDs to select data we're interested in
            stim_ids = stim_ids = np.sort(np.concatenate(stimIDs_correct_transitions))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = stimulus_df.loc[stim_ids]['image_id'].values
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'XA':
            
            #Get CX trials
            stimIDs_correct_transitions = []
            for iID in oddball_ids:
                #XA trials
                stimIDs =  stimulus_df[(stimulus_df.session_block_name == 'transition_control') & (stimulus_df.image_id == iID)].index.values
                for sID in stimIDs:
                    imgID = stimulus_df.loc[sID+1]['image_id']
                    #Get 'XA' A trials
                    if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'][0] == iID):
                        stimIDs_correct_transitions.append(sID) #Add X trial
                        stimIDs_correct_transitions.append(sID+1) #Add A trial
                        
            #Use presentation IDs to select data we're interested in
            stim_ids = stim_ids = np.sort(stimIDs_correct_transitions)
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = stimulus_df.loc[stim_ids]['image_id'].values
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        #Let's see if these 'A' trials are different
        elif trial_type == 'DAXA':
            #Get DA
            DA_stim_ids = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]):
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
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
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
            X_subset = X_responses.sel(stimulus_presentations_id = OB_stimIDs)
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
        elif trial_type == 'DX':
            
            #Get CD trials
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if imgID != mainseq_ids[-1]:
                    continue
                if transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']:
                    seq_ids_correct_transitions.append(sID)
            CD_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            CX_stimIDs  = np.array(seq_ids_correct_transitions)
            
            stim_ids = np.concatenate((CD_stimIDs,CX_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((np.repeat(0,len(CD_stimIDs)),np.repeat(1,len(CX_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'all':
            #Just get the oddball trials
            X_subset = X_responses
            stimIDs = X_subset.coords['stimulus_presentations_id'].values
            Y_subset = stimulus_df.loc[stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[stimIDs]['image_id'].values
            
#     elif block == 'occlusion':
#         if trial_type == 'X':
#             X_subset = X_responses.loc[OB_stimIDs]
#             Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
#             Y_sort = np.array(['{}/{}'.format(stimulus_df.loc[sID]['image_id'],stimulus_df.loc[sID]['fraction_occlusion']) for sID in OB_stimIDs])
            
    else:
        if trial_type == 'ABCD':
            #Just get the MS trials
            stimIDs = MS_stimIDs
            X_subset = X_responses.sel(stimulus_presentations_id = MS_stimIDs)
            Y_subset = stimulus_df.loc[MS_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[MS_stimIDs]['image_id'].values
            
        elif trial_type == 'X':
            #Just get the oddball trials
            X_subset = X_responses.sel(stimulus_presentations_id = OB_stimIDs)
            Y_subset = stimulus_df.loc[OB_stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[OB_stimIDs]['image_id'].values
            
        elif trial_type == 'ABCDX':
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))
            X_subset = X_responses.sel(stimulus_presentations_id = stim_ids)
            Y_subset = np.concatenate((stimulus_df.loc[MS_stimIDs]['image_id'],np.repeat(1,len(OB_stimIDs))))
            Y_sort = stimulus_df.loc[stim_ids]['image_id'].values
            
        elif trial_type == 'all':
            #Just get the oddball trials
            X_subset = X_responses
            stimIDs = X_subset.coords['stimulus_presentations_id'].values
            Y_subset = stimulus_df.loc[stimIDs]['image_id'].values
            Y_sort = stimulus_df.loc[stimIDs]['image_id'].values
    
    return X_subset, Y_subset, Y_sort

def get_trial_indices(stimulus_df, block='oddball', trial_type='ABCD', seq_dist=1, trial_indy=slice(None)):

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
            stim_ids = MS_stimIDs[trial_indy]
                                
        elif trial_type == 'ABCX':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,4)])) #just get the ABC before X
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((prevMS_stimIDs,OB_stimIDs))
            
        elif trial_type == 'DABCXABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,5)])) #just get the DABC before X
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.sort(np.concatenate((prevMS_stimIDs,OB_stimIDs,nextMS_stimIDs)))
            
            
        elif trial_type == 'ABCXABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,4)])) #just get the ABC before X
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.sort(np.concatenate((prevMS_stimIDs,OB_stimIDs,nextMS_stimIDs)))

#         elif trial_type == 'X-ABCD':

#             #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
#             prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,4)])) #just get the ABC before X
#             nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + ii for ii in range(1,5)])) #get the ABCD after X
            
#             #Use presentation IDs to select data we're interested in
#             stim_ids = np.sort(np.concatenate((OB_stimIDs,nextMS_stimIDs)))
            
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
            
        elif trial_type == 'ABCD-X':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs we want
            if seq_dist > 0:
                #We're taking ABCD---X
                prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - 4*seq_dist - ii for ii in range(4)]))
            elif seq_dist == 0:
                #We're taking DABCX, where D is shared above if seq_dist == 1
                prevMS_stimIDs = sorted(np.concatenate([OB_stimIDs - ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = prevMS_stimIDs
            
            
        elif trial_type == 'X-ABCD':

            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + 4*seq_dist + ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = nextMS_stimIDs
            

        elif trial_type == 'XABCD':

            #Use the oddball stimulus presentation IDs to identify the presentation IDs for the main sequence IDs that occur after an oddball has been presented
            nextMS_stimIDs = sorted(np.concatenate([OB_stimIDs + 4*seq_dist + ii for ii in range(1,5)]))

            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((nextMS_stimIDs,OB_stimIDs))
            
        elif trial_type == 'X':
            
            stim_ids  = OB_stimIDs

        elif trial_type == 'DAXA':
            #Get DA
            #Use the oddball stimulus presentation IDs to identify the main sequence image A that occurs after D but before X; DABCX
            DA_stimIDs = sorted(OB_stimIDs - 3 - 4*seq_dist)
            
            #And the A that occurs after X
            XA_stimIDs = sorted(OB_stimIDs + 1)
              
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((DA_stimIDs,XA_stimIDs))
            
        elif trial_type == 'DX':
            #Use the oddball stimulus presentation IDs to identify the main sequence image D before X; DABCX
            D_stimIDs = sorted(OB_stimIDs - 4*seq_dist)
            
            stim_ids = np.concatenate((D_stimIDs,OB_stimIDs))
            
        elif trial_type == 'XD':
            #Use the oddball stimulus presentation IDs to identify the main sequence image D after  X; XABCD
            D_stimIDs = sorted(OB_stimIDs + 4*seq_dist)
            
            stim_ids = np.concatenate((D_stimIDs,OB_stimIDs))            
            
    elif block == 'transition_control':
        #Make sure we take trials of the same transition type as in the oddball context
        #i.e. trials where image B was presented will be proceeded by tials where image A was shown
        if trial_type == 'ABCD': 
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            stim_ids = MS_stimIDs
        
        #Similarly, now we're going to include all of the DX transitions
        #In the transition context, ABCDX is almost the same as XABCD, except for which 'A' trials we want to include
        #ABCDX: (D,A),(A,B),(B,C),(C,D),(C,X)
        #XABCD: (X,A),(A,B),(B,C),(C,D),(C,X)
        elif trial_type == 'ABCDX':
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]:
                    seq_ids_correct_transitions.append(sID)
            MS_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))            

        #Similarly, now we're going to include all of the XA instead of DA transitions
        elif trial_type == 'XABCD':
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'][0] in oddball_ids):
                    seq_ids_correct_transitions.append(sID)
                #Get AB, BC, CD trials
                elif (imgID in mainseq_ids[1:]) & (stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]):
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
            
        #Similarly, now we're going to include all of the XA instead of DA transitions
        elif trial_type == 'CX':

            #Get CX trials
            stimIDs_correct_transitions = []
            for iID in oddball_ids:
                #CX trials
                stimIDs =  stimulus_df[(stimulus_df.session_block_name == 'transition_control') & (stimulus_df.image_id == iID)  & (stimulus_df.stimulus_key == transctrl_dict[iID])].index.values
                stimIDs_correct_transitions.append(stimIDs-1) #Append C trials
                stimIDs_correct_transitions.append(stimIDs) #Append X trials
                
            #Use presentation IDs to select data we're interested in
            stim_ids = stim_ids = np.sort(np.concatenate(stimIDs_correct_transitions))
            
        elif trial_type == 'XA':
            
            #Get CX trials
            stimIDs_correct_transitions = []
            for iID in oddball_ids:
                #XA trials
                stimIDs =  stimulus_df[(stimulus_df.session_block_name == 'transition_control') & (stimulus_df.image_id == iID)].index.values
                for sID in stimIDs:
                    imgID = stimulus_df.loc[sID+1]['image_id']
                    #Get 'XA' A trials
                    if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'][0] == iID):
                        stimIDs_correct_transitions.append(sID) #Add X trial
                        stimIDs_correct_transitions.append(sID+1) #Add A trial
                        
            #Use presentation IDs to select data we're interested in
            stim_ids = stim_ids = np.sort(stimIDs_correct_transitions)
            
        #Let's see if these 'A' trials are different
        elif trial_type == 'DAXA':
            #Get DA
            DA_stim_ids = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                #Get 'XA' A trials
                if (imgID == mainseq_ids[0]) & (stimulus_df.loc[sID]['stimulus_key'] == transctrl_dict[imgID]):
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
            
        elif trial_type == 'X':
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            OB_stimIDs  = np.array(seq_ids_correct_transitions)
            stim_ids = OB_stimIDs

        elif trial_type == 'DX':
            
            #Get CD trials
            seq_ids_correct_transitions = []
            for sID in MS_stimIDs:
                imgID = stimulus_df.loc[sID]['image_id']
                if imgID != mainseq_ids[-1]:
                    continue
                if transctrl_dict[imgID] == stimulus_df.loc[sID]['stimulus_key']:
                    seq_ids_correct_transitions.append(sID)
            CD_stimIDs = np.array(seq_ids_correct_transitions)
            
            #Get CX trials
            seq_ids_correct_transitions = []
            for sID in OB_stimIDs:
                if stimulus_df.loc[sID]['stimulus_key'][0] == mainseq_ids[-2]:
                    seq_ids_correct_transitions.append(sID)
            CX_stimIDs  = np.array(seq_ids_correct_transitions)
            stim_ids = np.concatenate((CD_stimIDs,CX_stimIDs))
            
    else:
        if trial_type == 'ABCD':
            #Just get the MS trials
            stim_ids = MS_stimIDs
            
        elif trial_type == 'X':
            stim_ids = OB_stimIDs

        elif trial_type == 'ABCDX':
            #Use presentation IDs to select data we're interested in
            stim_ids = np.concatenate((MS_stimIDs,OB_stimIDs))
    
    return stim_ids

##==================================================
def decode_labels(X,Y,train_index,test_index,classifier='LDA',clabels=None,X_test=None,Y_test=None,shuffle=True,parallel=True,classifier_kws=None):
    
    nTrials, nNeurons = X.shape

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
        train_index = np.arange(len(X_train))
#         print('.',end='')
        
    #Copy training index for shuffle decoding
    train_index_sh = train_index.copy()

     
    #How many classes are we trying to classify?
    class_labels,nTrials_class = np.unique(Y,return_counts=True)
    nClasses = len(class_labels)

    #Initialize Classifier
    if classifier == 'LDA':
#         clf = LinearDiscriminantAnalysis()
        clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    elif classifier == 'SVM':
        clf = svm.LinearSVC(max_iter=1E6) #penalty='l1',dual=False,
    elif classifier == 'NLSVM':
        clf = svm.NuSVC(gamma="auto")
    elif classifier == 'QDA':
        clf = QuadraticDiscriminantAnalysis()
    elif classifier == 'NearestNeighbors':
        clf = KNeighborsClassifier()
    elif classifier == 'LinearSVM':
        clf = SVC(kernel="linear", C=0.025)
    elif classifier == 'RBFSVM':
        clf = SVC(gamma=2, C=1)
    elif classifier == 'GP':
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif classifier == 'DecisionTree':
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'RandomForest':
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif classifier == 'NeuralNet':
        clf = MLPClassifier(alpha=1, max_iter=1000)
    elif classifier == 'NaiveBayes':
        clf = GaussianNB()
    
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

    #All other classifiers
    else:
        #Fit model to the training data
        clf.fit(X_train, Y_train)

        #Predict test data
        Y_hat = clf.predict(X_test)
        Y_hat_train = clf.predict(X_train)

    #Calculate confusion matrix
    kfold_hits = confusion_matrix(Y_test,Y_hat,labels=clabels)
#     pdb.set_trace()
    ##===== Perform Shuffle decoding =====##
    if shuffle:
        kfold_shf = np.zeros((nShuffles,nClasses,nClasses))
        
        if parallel:
            with Pool(50) as p:
                processes = []
                #Classify with shuffled dataset
                for iS in range(nShuffles):
                    np.random.shuffle(train_index_sh)
                    Y_train_sh = Y[train_index_sh]

                    processes.append(p.apply_async(decode_labels,args=(X_train,Y_train_sh,None,None,classifier,clabels,X_test,Y_test,False,None)))

                #Extract results from parallel kfold processing
                kfold_shf = np.array([p.get()[0] for p in processes])
        else:
            
            kfold_shf_list = []
            for iS in range(nShuffles):
                np.random.shuffle(train_index_sh)
                Y_train_sh = Y[train_index_sh]
                kfshf, _ = decode_labels(X_train,Y_train_sh,None,None,classifier,clabels,X_test,Y_test,False,None)
                kfold_shf_list.append(kfshf)
            kfold_shf = np.array(kfold_shf_list)
                
            
    else:
        kfold_shf = np.ones((nShuffles,nClasses,nClasses))*(1/nClasses)

#     pdb.set_trace()
    return kfold_hits, kfold_shf #, decoding_weights, decoding_weights_z, decoding_weights_m_shf, decoding_weights_s_shf

##==================================================
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
        confusion_z =  (confusion_mat - m_shf)/s_shf

#         pdb.set_trace()
#         #Get signficance of decoding 
#         pvalues_loo = st.norm.sf(confusion_z)
        
        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        N = confusion_mat.shape[0]
        pvalues = 1-2*np.abs(np.array([[st.percentileofscore(c_shf[:,i,j],confusion_mat[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        
        if plot_shuffle:
            #Plot shuffle distributions
            title = 'Leave-1-Out Cross-Validation'
            plot_decoding_shuffle(confusion_mat, c_shf, pvalues_loo, title,pdfdoc)
            
    elif method == 'kfold':       
        kfold_accuracies = []
        shf_accuracies = []
        kfold_zscores = []
        kfold_pvalues = []
#         pdb.set_trace()
        #Calculate decoding accuracy per kfold
        for iK,rTuple in enumerate(results):
            kfold_hits = rTuple[0] #size [nClasses x nClasses]
            kfold_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]

            #Normalize confusion matrix
            cm = kfold_hits/np.sum(kfold_hits,axis=1).reshape(-1,1)
            kfold_accuracies.append(cm)
            
            #Loop through shuffles and normalize
            c_shf = np.zeros((nShuffles,nClasses,nClasses))
            for iS in range(nShuffles):
                c_shf[iS] = kfold_shf[iS]/np.sum(kfold_shf[iS],axis=1).reshape(-1,1)

            #Calculate z-score for this kfold
            m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
            shf_accuracies.append(m_shf)
            kfold_z = np.divide(kfold_accuracies[iK] - m_shf,s_shf,np.zeros(kfold_hits.shape),where=s_shf!=0)
            kfold_zscores.append(kfold_z)

#             #Get signficance of decoding 
#             pvalues_kfold = st.norm.sf(kfold_z)
#             pdb.set_trace()

            #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
            N = cm.shape[0]
            pvalues_kfold = 1-2*np.abs(np.array([[st.percentileofscore(c_shf[:,i,j],cm[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
            kfold_pvalues.append(pvalues_kfold)
        
            if plot_shuffle:
                #Plot shuffle distributions
                title = 'Shuffle Distributions for kfold {}'.format(iK)
                plot_decoding_shuffle(kfold_accuracies[iK], c_shf, pvalues_kfold, title,pdfdoc)

        #Take average over kfolds
        confusion_mat = np.mean(kfold_accuracies,axis=0)
        confusion_shf = np.mean(shf_accuracies,axis=0)
        confusion_z = np.mean(kfold_zscores,axis=0)
        # stdevs = (np.std(kfold_accuracies,axis=0),np.std(kfold_zscores,axis=0))
        pvalues = np.mean(kfold_pvalues,axis=0)
        
    return confusion_mat, confusion_shf, confusion_z, pvalues
    
##==================================================
def cross_validate(X,Y,Y_sort=None,method='L1O',nKfold=5,classifier='SVM',classifier_kws=None,clabels=None,shuffle=True,plot_shuffle=False,parallel=False,nProcesses=30,pdfdoc=None):
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
        # print(np.unique(Y[train_index],return_counts=True))
        # print(np.unique(Y_sort[train_index],return_counts=True))
#         pdb.set_trace()
        if parallel:
            processes.append(pool.apply_async(decode_labels,args=(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,False,classifier_kws)))
        else:
            # print(f'\nkfold {iK} - ')
            tmp = decode_labels(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,False,classifier_kws)
            results.append(tmp)
    # pdb.set_trace()
    #Extract results from parallel kfold processing
    if parallel:
        results = [p.get() for p in processes]
        pool.close()
        
    ##===== Calculate decoding accuracy =====##
    confusion_mat, confusion_shf, confusion_z, pvalues = calculate_accuracy(results,method,plot_shuffle,pdfdoc)
#     pdb.set_trace()
    return confusion_mat, confusion_shf, confusion_z, pvalues


##==============================##
##===== Plotting Functions =====##

##==================================================
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
    
##==================================================
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

##==================================================
def plot_decoding_accuracy(confusion_mat,confusion_z,ax=None,block='randomized_ctrl',class_labels=None,xylabels=None,title=None,annot=True,cmap='rocket',clims=None,cbar=True,sigfig=True,pdfdoc=None,shrink=0.5):
    #Plot decoding performance
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))#,
        
    if title is not None:
        ax.set_title(title,fontsize=16)

    pvalues = st.norm.sf(confusion_z)
    
    if clims is None:
        clims = [np.percentile(confusion_mat,1) % 0.05, np.percentile(confusion_mat,99) - np.percentile(confusion_mat,99) % 0.05]
    #Plot actual decoding performance
    sns.heatmap(confusion_mat,annot=annot,fmt='2.2f',annot_kws={'fontsize': 10},cbar=cbar,square=True,cmap=cmap,vmin=clims[0],vmax=clims[1],cbar_kws={'shrink': shrink,'ticks':clims,'label': 'Accuracy'},ax=ax,rasterized=True)

    if sigfig:
        pval = (pvalues < 0.05) #& np.eye(pvalues.shape[0],dtype=bool)
        x = np.linspace(0, pval.shape[0]-1, pval.shape[0])+0.5
        y = np.linspace(0, pval.shape[1]-1, pval.shape[1])+0.5
        X, Y = np.meshgrid(x, y)
        if len(pval) > 5:
            ax.scatter(X,Y,s=10*pval, marker='.',c='k')
        else:
            ax.scatter(X,Y,s=35*pval, marker='.',c='k')
            
#     if xylabels is not None:
#         #Labels
#         ax.set_ylabel(xylabels[0],fontsize=16)
#         ax.set_xlabel(xylabels[1],fontsize=16)
#     else:
#         ax.set_ylabel('Actual Image',fontsize=16)
#         ax.set_xlabel('Decoded Image',fontsize=16)

    if class_labels is not None:
        # ax.set_yticks(np.arange(len(class_labels))+0.5)
        # ax.set_xticks(np.arange(len(class_labels))+0.5) 
        ax.set_yticklabels(class_labels)#,va="center",fontsize=14)
        ax.set_xticklabels(class_labels)#,rotation=45)#,va="center",fontsize=14) 
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        
        
def plot_confusion_matrices(confusion_mat,confusion_z,ax=None,plot_titles=['A','B','C','D'],class_labels=['rand-pre','OB', 'Trans','rand-post']):
                                       
    nClasses = confusion_mat.shape[0]
    fig, axes = plt.subplots(1,4,figsize=(12,3))
    for i in range(nClasses):
        ax = axes[i]
        
        plot_decoding_accuracy(confusion_mat[i],confusion_z[i],ax=ax,class_labels=class_labels,xylabels=None,title=plot_titles[i],annot=False,clims=[0,1],cbar=True,sigfig=True,pdfdoc=None,shrink=0.5)
        
    return fig

        
        
                                       
                
