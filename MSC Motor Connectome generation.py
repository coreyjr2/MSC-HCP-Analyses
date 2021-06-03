# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nilearn as nl
import numpy as np
import nibabel as nib
import pandas as pd
import sklearn


#Directories and session/subject lists for looping over
DATA_DIR = '/home/corey/Midnight/Data/'
func_file_path = '{DATA_DIR}/{Subject_list}/func/{file_name}'
Subject_List = ['sub-MSC01', 'sub-MSC02', 'sub-MSC03', 'sub-MSC04', 'sub-MSC05', 
                'sub-MSC06', 'sub-MSC07', 'sub-MSC08', 'sub-MSC09', 'sub-MSC10' ]

Session_list = ['ses-func01', 'ses-func02', 'ses-func03', 'ses-func04', 'ses-func05',
                'ses-func06', 'ses-func07', 'ses-func08', 'ses-func09', 'ses-func10']
Run_list = ['01', '02']

block_list = ['RHand', "RFoot", 'LHand', 'LFoot','Tongue']



#np.empty(n_subj, 5, 4560)
#make_timeseries_to_vector('/home/corey/Midnight/Data/{subject}/{session}/func/{subject}_{session}_task-motor_{run}_bold.nii.gz')


def make_timeseries_to_vector(subject, session, run):
    '''This function takes raw timeseries from a participant's motor task scan and transforms it
    into a series of functional connectivity matrices associated with each condition (that is, a block 
    for each limb movement)'''
    data_path_template = f'/home/corey/Midnight/Data/{subject}/{session}/func/{subject}_{session}_task-motor_run-{run}_bold.nii.gz'
    #load in raw timeseries
    raw_timeseries = nib.load(data_path_template)
    
    #Importhe atlas and the masker 
    from nilearn import datasets
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    atlas_filename = dataset.maps
    labels = dataset.labels
    from nilearn import plotting
    plotting.plot_roi(atlas_filename)
    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    
    
    #Transform the motor task imaging data with the masker and check the shape
    masked_timeseries = []
    masked_timeseries = masker.fit_transform(raw_timeseries)
    
    
    #Load in the envent files, and convert the durations from seconds to TRs
    motor_tsv = pd.read_csv(f'/home/corey/Midnight/Data/{subject}/{session}/func/{subject}_{session}_task-motor_run-{run}_events.tsv', sep = '\t')
    motor_tsv['onset'] = motor_tsv['onset']/2.2  
    motor_tsv['duration'] = motor_tsv['duration']/2.2  
    
    
   # Arrange the time series from this subject's run
    mapper = dict([((name, i), list(range(int(unique_val), int(unique_val+7)))) for i, (unique_val, name) in 
                                    enumerate(zip(motor_tsv['onset'], motor_tsv['trial_type']))])
    from collections import defaultdict
    mapper_updated = defaultdict(list)
    for name_tup, TRs in mapper.items():
      mapper_updated[name_tup[0]].extend(TRs)
    
    
    #Assemble all of the time series into a single object
    block_timeseries = {}
    for name, TRs in mapper_updated.items():
        block_timeseries[name] = masked_timeseries[[int(TR) for TR in TRs]]
        
      
        
    #Calculate FC for each timeseries
    #import relevant libraries
    from nilearn.connectome import sym_matrix_to_vec
    from nilearn.connectome import ConnectivityMeasure
    
    #create an object for calculating FC
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    #initialize an object for the subject's FC matrices for each condition 
    task_fc_matrix = np.zeros((5,96,96))
    
    
    task_fc_matrix[:, :, :] = correlation_measure.fit_transform(block_timeseries.values())
    
    #length of vector is 4512
    vector_motor = np.zeros((5,4560))
    
    #extract the lower diagonal values and place them in their own matrix
    for i in range(task_fc_matrix.shape[0]):
        #vector_motor[subject,:] = sym_matrix_to_vec(task_fc_matrix[i,:,:], discard_diagonal=True)
        vector_motor[i,:] = task_fc_matrix[i][np.triu_indices_from(task_fc_matrix[i], k=1)]
    
    #make the data a pandas dataframe
    vector_motor = pd.DataFrame(vector_motor)
   
    #Now we need to attach information about what list of connectivity values corresponds to which movement
    movement_col = ["LFoot","LHand", "RFoot", "RHand", "Tongue" ]
    vector_motor.insert(0, "Block", movement_col)
    
    #Now we insert other information into the dataframe
    vector_motor.insert(0, "Subject ID", subject)
    vector_motor.insert(1, "Session", session)
    vector_motor.insert(2, "Run", run)
    return vector_motor
    
    
    
    
 
    



'''This is long and annoying, but testing to make sure every singel data 
point loads coorectly within the function'''


#test out function on individual datasets for subject 1
msc_01_01_01 = make_timeseries_to_vector('sub-MSC01','ses-func01', '01')
msc_01_01_02 = make_timeseries_to_vector('sub-MSC01','ses-func01', '02')
msc_01_02_01 = make_timeseries_to_vector('sub-MSC01','ses-func02', '01')
msc_01_02_02 = make_timeseries_to_vector('sub-MSC01','ses-func02', '02')
msc_01_03_01 = make_timeseries_to_vector('sub-MSC01','ses-func03', '01')
msc_01_03_02 = make_timeseries_to_vector('sub-MSC01','ses-func03', '02')
msc_01_04_01 = make_timeseries_to_vector('sub-MSC01','ses-func04', '01')
msc_01_04_02 = make_timeseries_to_vector('sub-MSC01','ses-func04', '02')
msc_01_05_01 = make_timeseries_to_vector('sub-MSC01','ses-func05', '01')
msc_01_05_02 = make_timeseries_to_vector('sub-MSC01','ses-func05', '02')
msc_01_06_01 = make_timeseries_to_vector('sub-MSC01','ses-func06', '01')
msc_01_06_02 = make_timeseries_to_vector('sub-MSC01','ses-func06', '02')
msc_01_07_01 = make_timeseries_to_vector('sub-MSC01','ses-func07', '01')
msc_01_07_02 = make_timeseries_to_vector('sub-MSC01','ses-func07', '02')
msc_01_08_01 = make_timeseries_to_vector('sub-MSC01','ses-func08', '01')
msc_01_08_02 = make_timeseries_to_vector('sub-MSC01','ses-func08', '02')
msc_01_09_01 = make_timeseries_to_vector('sub-MSC01','ses-func09', '01')
msc_01_09_02 = make_timeseries_to_vector('sub-MSC01','ses-func09', '02')
msc_01_10_01 = make_timeseries_to_vector('sub-MSC01','ses-func10', '01')
msc_01_10_02 = make_timeseries_to_vector('sub-MSC01','ses-func10', '02')

#concatenate them all
msc_01_data = pd.concat([msc_01_01_01, msc_01_01_02, msc_01_02_01, msc_01_02_02, 
                         msc_01_03_01, msc_01_03_02, msc_01_04_01, msc_01_04_02,
                         msc_01_05_01, msc_01_05_02, msc_01_06_01, msc_01_06_02,
                         msc_01_07_01, msc_01_07_02, msc_01_08_01, msc_01_08_02,
                         msc_01_09_01, msc_01_09_02, msc_01_10_01, msc_01_10_02])



#resent the index
msc_01_data = msc_01_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_01_data.rename(columns={'index':'Block (numeric)'}, inplace=True)


###############################################################################

#test out function on individual datasets for subject 2
msc_02_01_01 = make_timeseries_to_vector('sub-MSC02','ses-func01', '01')
msc_02_01_02 = make_timeseries_to_vector('sub-MSC02','ses-func01', '02')
msc_02_02_01 = make_timeseries_to_vector('sub-MSC02','ses-func02', '01')
msc_02_02_02 = make_timeseries_to_vector('sub-MSC02','ses-func02', '02')
msc_02_03_01 = make_timeseries_to_vector('sub-MSC02','ses-func03', '01')
#msc_02_03_02 = make_timeseries_to_vector('sub-MSC02','ses-func03', '02') #use this one to check shape because we are having problems with the others
msc_02_03_02_path = f'/home/corey/Midnight/Data/sub-MSC02/ses-func03/func/sub-MSC02_ses-func03_task-motor_run-02_bold.nii.gz'
nib.load(msc_02_03_02_path).shape #default size seems to be 64x64x36x104
#msc_02_04_01 = make_timeseries_to_vector('sub-MSC02','ses-func04', '01') #This one has a weird shape
msc_02_04_01_path = f'/home/corey/Midnight/Data/sub-MSC02/ses-func04/func/sub-MSC02_ses-func04_task-motor_run-01_bold.nii.gz'
nib.load(msc_02_04_01_path).shape #weird, they are the same size
msc_02_04_02 = make_timeseries_to_vector('sub-MSC02','ses-func04', '02') # so does this one
msc_02_05_01 = make_timeseries_to_vector('sub-MSC02','ses-func05', '01')
msc_02_05_02 = make_timeseries_to_vector('sub-MSC02','ses-func05', '02')
msc_02_06_01 = make_timeseries_to_vector('sub-MSC02','ses-func06', '01')
msc_02_06_02 = make_timeseries_to_vector('sub-MSC02','ses-func06', '02')
#msc_02_07_01 = make_timeseries_to_vector('sub-MSC02','ses-func07', '01') #this one is also a weird size
#msc_02_07_02 = make_timeseries_to_vector('sub-MSC02','ses-func07', '02') #as is this one
msc_02_08_01 = make_timeseries_to_vector('sub-MSC02','ses-func08', '01')
msc_02_08_02 = make_timeseries_to_vector('sub-MSC02','ses-func08', '02')
msc_02_09_01 = make_timeseries_to_vector('sub-MSC02','ses-func09', '01')
msc_02_09_02 = make_timeseries_to_vector('sub-MSC02','ses-func09', '02')
msc_02_10_01 = make_timeseries_to_vector('sub-MSC02','ses-func10', '01')
msc_02_10_02 = make_timeseries_to_vector('sub-MSC02','ses-func10', '02')

#concatenate them all
msc_02_data = pd.concat([msc_02_01_01, msc_02_01_02, msc_02_02_01, msc_02_02_02, 
                         msc_02_03_01, msc_02_03_02,
                         msc_02_05_01, msc_02_05_02, msc_02_06_01, msc_02_06_02,
                                                     msc_02_08_01, msc_02_08_02,
                         msc_02_09_01, msc_02_09_02, msc_02_10_01, msc_02_10_02])



#resent the index
msc_02_data = msc_02_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_02_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 3
msc_03_01_01 = make_timeseries_to_vector('sub-MSC03','ses-func01', '01')
msc_03_01_02 = make_timeseries_to_vector('sub-MSC03','ses-func01', '02')
msc_03_02_01 = make_timeseries_to_vector('sub-MSC03','ses-func02', '01')
msc_03_02_02 = make_timeseries_to_vector('sub-MSC03','ses-func02', '02')
msc_03_03_01 = make_timeseries_to_vector('sub-MSC03','ses-func03', '01')
msc_03_03_02 = make_timeseries_to_vector('sub-MSC03','ses-func03', '02')
msc_03_04_01 = make_timeseries_to_vector('sub-MSC03','ses-func04', '01')
msc_03_04_02 = make_timeseries_to_vector('sub-MSC03','ses-func04', '02')
msc_03_05_01 = make_timeseries_to_vector('sub-MSC03','ses-func05', '01')
msc_03_05_02 = make_timeseries_to_vector('sub-MSC03','ses-func05', '02')
msc_03_06_01 = make_timeseries_to_vector('sub-MSC03','ses-func06', '01')
msc_03_06_02 = make_timeseries_to_vector('sub-MSC03','ses-func06', '02')
msc_03_07_01 = make_timeseries_to_vector('sub-MSC03','ses-func07', '01')
msc_03_07_02 = make_timeseries_to_vector('sub-MSC03','ses-func07', '02')
msc_03_08_01 = make_timeseries_to_vector('sub-MSC03','ses-func08', '01')
msc_03_08_02 = make_timeseries_to_vector('sub-MSC03','ses-func08', '02')
msc_03_09_01 = make_timeseries_to_vector('sub-MSC03','ses-func09', '01')
msc_03_09_02 = make_timeseries_to_vector('sub-MSC03','ses-func09', '02')
msc_03_10_01 = make_timeseries_to_vector('sub-MSC03','ses-func10', '01')
msc_03_10_02 = make_timeseries_to_vector('sub-MSC03','ses-func10', '02')

#concatenate them all
msc_03_data = pd.concat([msc_03_01_01, msc_03_01_02, msc_03_02_01, msc_03_02_02, 
                         msc_03_03_01, msc_03_03_02, msc_03_04_01, msc_03_04_02,
                         msc_03_05_01, msc_03_05_02, msc_03_06_01, msc_03_06_02,
                         msc_03_07_01, msc_03_07_02, msc_03_08_01, msc_03_08_02,
                         msc_03_09_01, msc_03_09_02, msc_03_10_01, msc_03_10_02])


#resent the index
msc_03_data = msc_03_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_03_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 4
msc_04_01_01 = make_timeseries_to_vector('sub-MSC04','ses-func01', '01')
msc_04_01_02 = make_timeseries_to_vector('sub-MSC04','ses-func01', '02')
msc_04_02_01 = make_timeseries_to_vector('sub-MSC04','ses-func02', '01')
msc_04_02_02 = make_timeseries_to_vector('sub-MSC04','ses-func02', '02')
msc_04_03_01 = make_timeseries_to_vector('sub-MSC04','ses-func03', '01')
msc_04_03_02 = make_timeseries_to_vector('sub-MSC04','ses-func03', '02')
msc_04_04_01 = make_timeseries_to_vector('sub-MSC04','ses-func04', '01')
msc_04_04_02 = make_timeseries_to_vector('sub-MSC04','ses-func04', '02')
msc_04_05_01 = make_timeseries_to_vector('sub-MSC04','ses-func05', '01')
msc_04_05_02 = make_timeseries_to_vector('sub-MSC04','ses-func05', '02')
#msc_04_06_01 = make_timeseries_to_vector('sub-MSC04','ses-func06', '01') #These two have weird shapes
#msc_04_06_02 = make_timeseries_to_vector('sub-MSC04','ses-func06', '02') #These two have weird shapes
msc_04_07_01 = make_timeseries_to_vector('sub-MSC04','ses-func07', '01')
msc_04_07_02 = make_timeseries_to_vector('sub-MSC04','ses-func07', '02')
msc_04_08_01 = make_timeseries_to_vector('sub-MSC04','ses-func08', '01')
msc_04_08_02 = make_timeseries_to_vector('sub-MSC04','ses-func08', '02')
msc_04_09_01 = make_timeseries_to_vector('sub-MSC04','ses-func09', '01')
msc_04_09_02 = make_timeseries_to_vector('sub-MSC04','ses-func09', '02')
#msc_04_10_01 = make_timeseries_to_vector('sub-MSC04','ses-func10', '01') #These two have weird shapes
#msc_04_10_02 = make_timeseries_to_vector('sub-MSC04','ses-func10', '02') #These two have weird shapes

#concatenate them all
msc_04_data = pd.concat([msc_04_01_01, msc_04_01_02, msc_04_02_01, msc_04_02_02, 
                         msc_04_03_01, msc_04_03_02, msc_04_04_01, msc_04_04_02,
                         msc_04_05_01, msc_04_05_02, 
                         msc_04_07_01, msc_04_07_02, msc_04_08_01, msc_04_08_02,
                         msc_04_09_01, msc_04_09_02,                           ])


#resent the index
msc_04_data = msc_04_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_04_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################

#test out function on individual datasets for subject 5
msc_05_01_01 = make_timeseries_to_vector('sub-MSC05','ses-func01', '01')
msc_05_01_02 = make_timeseries_to_vector('sub-MSC05','ses-func01', '02')
msc_05_02_01 = make_timeseries_to_vector('sub-MSC05','ses-func02', '01')
msc_05_02_02 = make_timeseries_to_vector('sub-MSC05','ses-func02', '02')
msc_05_03_01 = make_timeseries_to_vector('sub-MSC05','ses-func03', '01')
msc_05_03_02 = make_timeseries_to_vector('sub-MSC05','ses-func03', '02')
msc_05_04_01 = make_timeseries_to_vector('sub-MSC05','ses-func04', '01')
msc_05_04_02 = make_timeseries_to_vector('sub-MSC05','ses-func04', '02')
msc_05_05_01 = make_timeseries_to_vector('sub-MSC05','ses-func05', '01')
msc_05_05_02 = make_timeseries_to_vector('sub-MSC05','ses-func05', '02')
msc_05_06_01 = make_timeseries_to_vector('sub-MSC05','ses-func06', '01')
msc_05_06_02 = make_timeseries_to_vector('sub-MSC05','ses-func06', '02')
msc_05_07_01 = make_timeseries_to_vector('sub-MSC05','ses-func07', '01')
msc_05_07_02 = make_timeseries_to_vector('sub-MSC05','ses-func07', '02')
msc_05_08_01 = make_timeseries_to_vector('sub-MSC05','ses-func08', '01')
msc_05_08_02 = make_timeseries_to_vector('sub-MSC05','ses-func08', '02')
msc_05_09_01 = make_timeseries_to_vector('sub-MSC05','ses-func09', '01')
msc_05_09_02 = make_timeseries_to_vector('sub-MSC05','ses-func09', '02')
msc_05_10_01 = make_timeseries_to_vector('sub-MSC05','ses-func10', '01')
msc_05_10_02 = make_timeseries_to_vector('sub-MSC05','ses-func10', '02')

#concatenate them all
msc_05_data = pd.concat([msc_05_01_01, msc_05_01_02, msc_05_02_01, msc_05_02_02, 
                         msc_05_03_01, msc_05_03_02, msc_05_04_01, msc_05_04_02,
                         msc_05_05_01, msc_05_05_02, msc_05_06_01, msc_05_06_02,
                         msc_05_07_01, msc_05_07_02, msc_05_08_01, msc_05_08_02,
                         msc_05_09_01, msc_05_09_02, msc_05_10_01, msc_05_10_02])

#resent the index
msc_05_data = msc_05_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_05_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 6
msc_06_01_01 = make_timeseries_to_vector('sub-MSC06','ses-func01', '01')
msc_06_01_02 = make_timeseries_to_vector('sub-MSC06','ses-func01', '02')
msc_06_02_01 = make_timeseries_to_vector('sub-MSC06','ses-func02', '01')
msc_06_02_02 = make_timeseries_to_vector('sub-MSC06','ses-func02', '02')
msc_06_03_01 = make_timeseries_to_vector('sub-MSC06','ses-func03', '01')
msc_06_03_02 = make_timeseries_to_vector('sub-MSC06','ses-func03', '02')
msc_06_04_01 = make_timeseries_to_vector('sub-MSC06','ses-func04', '01')
msc_06_04_02 = make_timeseries_to_vector('sub-MSC06','ses-func04', '02')
msc_06_05_01 = make_timeseries_to_vector('sub-MSC06','ses-func05', '01')
msc_06_05_02 = make_timeseries_to_vector('sub-MSC06','ses-func05', '02')
msc_06_06_01 = make_timeseries_to_vector('sub-MSC06','ses-func06', '01')
msc_06_06_02 = make_timeseries_to_vector('sub-MSC06','ses-func06', '02')
msc_06_07_01 = make_timeseries_to_vector('sub-MSC06','ses-func07', '01')
msc_06_07_02 = make_timeseries_to_vector('sub-MSC06','ses-func07', '02')
msc_06_08_01 = make_timeseries_to_vector('sub-MSC06','ses-func08', '01')
msc_06_08_02 = make_timeseries_to_vector('sub-MSC06','ses-func08', '02')
msc_06_09_01 = make_timeseries_to_vector('sub-MSC06','ses-func09', '01')
msc_06_09_02 = make_timeseries_to_vector('sub-MSC06','ses-func09', '02')
msc_06_10_01 = make_timeseries_to_vector('sub-MSC06','ses-func10', '01')
msc_06_10_02 = make_timeseries_to_vector('sub-MSC06','ses-func10', '02')

#concatenate them all
msc_06_data = pd.concat([msc_06_01_01, msc_06_01_02, msc_06_02_01, msc_06_02_02, 
                         msc_06_03_01, msc_06_03_02, msc_06_04_01, msc_06_04_02,
                         msc_06_05_01, msc_06_05_02, msc_06_06_01, msc_06_06_02,
                         msc_06_07_01, msc_06_07_02, msc_06_08_01, msc_06_08_02,
                         msc_06_09_01, msc_06_09_02, msc_06_10_01, msc_06_10_02])

#resent the index
msc_06_data = msc_06_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_06_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 7
msc_07_01_01 = make_timeseries_to_vector('sub-MSC07','ses-func01', '01')
msc_07_01_02 = make_timeseries_to_vector('sub-MSC07','ses-func01', '02')
msc_07_02_01 = make_timeseries_to_vector('sub-MSC07','ses-func02', '01')
msc_07_02_02 = make_timeseries_to_vector('sub-MSC07','ses-func02', '02')
msc_07_03_01 = make_timeseries_to_vector('sub-MSC07','ses-func03', '01')
msc_07_03_02 = make_timeseries_to_vector('sub-MSC07','ses-func03', '02')
msc_07_04_01 = make_timeseries_to_vector('sub-MSC07','ses-func04', '01')
msc_07_04_02 = make_timeseries_to_vector('sub-MSC07','ses-func04', '02')
msc_07_05_01 = make_timeseries_to_vector('sub-MSC07','ses-func05', '01')
msc_07_05_02 = make_timeseries_to_vector('sub-MSC07','ses-func05', '02')
msc_07_06_01 = make_timeseries_to_vector('sub-MSC07','ses-func06', '01')
msc_07_06_02 = make_timeseries_to_vector('sub-MSC07','ses-func06', '02')
msc_07_07_01 = make_timeseries_to_vector('sub-MSC07','ses-func07', '01')
msc_07_07_02 = make_timeseries_to_vector('sub-MSC07','ses-func07', '02')
msc_07_08_01 = make_timeseries_to_vector('sub-MSC07','ses-func08', '01')
msc_07_08_02 = make_timeseries_to_vector('sub-MSC07','ses-func08', '02')
msc_07_09_01 = make_timeseries_to_vector('sub-MSC07','ses-func09', '01')
msc_07_09_02 = make_timeseries_to_vector('sub-MSC07','ses-func09', '02')
msc_07_10_01 = make_timeseries_to_vector('sub-MSC07','ses-func10', '01')
msc_07_10_02 = make_timeseries_to_vector('sub-MSC07','ses-func10', '02')
#concatenate them all
msc_07_data = pd.concat([msc_07_01_01, msc_07_01_02, msc_07_02_01, msc_07_02_02, 
                         msc_07_03_01, msc_07_03_02, msc_07_04_01, msc_07_04_02,
                         msc_07_05_01, msc_07_05_02, msc_07_06_01, msc_07_06_02,
                         msc_07_07_01, msc_07_07_02, msc_07_08_01, msc_07_08_02,
                         msc_07_09_01, msc_07_09_02, msc_07_10_01, msc_07_10_02])

#resent the index
msc_07_data = msc_07_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_07_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 9
msc_09_01_01 = make_timeseries_to_vector('sub-MSC09','ses-func01', '01')
msc_09_01_02 = make_timeseries_to_vector('sub-MSC09','ses-func01', '02')
msc_09_02_01 = make_timeseries_to_vector('sub-MSC09','ses-func02', '01')
msc_09_02_02 = make_timeseries_to_vector('sub-MSC09','ses-func02', '02')
msc_09_03_01 = make_timeseries_to_vector('sub-MSC09','ses-func03', '01')
msc_09_03_02 = make_timeseries_to_vector('sub-MSC09','ses-func03', '02')
msc_09_04_01 = make_timeseries_to_vector('sub-MSC09','ses-func04', '01')
msc_09_04_02 = make_timeseries_to_vector('sub-MSC09','ses-func04', '02')
msc_09_05_01 = make_timeseries_to_vector('sub-MSC09','ses-func05', '01')
msc_09_05_02 = make_timeseries_to_vector('sub-MSC09','ses-func05', '02')
msc_09_06_01 = make_timeseries_to_vector('sub-MSC09','ses-func06', '01')
msc_09_06_02 = make_timeseries_to_vector('sub-MSC09','ses-func06', '02')
msc_09_07_01 = make_timeseries_to_vector('sub-MSC09','ses-func07', '01')
msc_09_07_02 = make_timeseries_to_vector('sub-MSC09','ses-func07', '02')
msc_09_08_01 = make_timeseries_to_vector('sub-MSC09','ses-func08', '01')
msc_09_08_02 = make_timeseries_to_vector('sub-MSC09','ses-func08', '02')
msc_09_09_01 = make_timeseries_to_vector('sub-MSC09','ses-func09', '01')
msc_09_09_02 = make_timeseries_to_vector('sub-MSC09','ses-func09', '02')
msc_09_10_01 = make_timeseries_to_vector('sub-MSC09','ses-func10', '01')
msc_09_10_02 = make_timeseries_to_vector('sub-MSC09','ses-func10', '02')
#concatenate them all
msc_09_data = pd.concat([msc_09_01_01, msc_09_01_02, msc_09_02_01, msc_09_02_02, 
                         msc_09_03_01, msc_09_03_02, msc_09_04_01, msc_09_04_02,
                         msc_09_05_01, msc_08_05_02, msc_09_06_01, msc_09_06_02,
                         msc_09_07_01, msc_09_07_02, msc_09_08_01, msc_09_08_02,
                         msc_09_09_01, msc_09_09_02, msc_09_10_01, msc_09_10_02])
#resent the index
msc_09_data = msc_09_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_09_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 8
msc_08_01_01 = make_timeseries_to_vector('sub-MSC08','ses-func01', '01')
msc_08_01_02 = make_timeseries_to_vector('sub-MSC08','ses-func01', '02')
msc_08_02_01 = make_timeseries_to_vector('sub-MSC08','ses-func02', '01')
msc_08_02_02 = make_timeseries_to_vector('sub-MSC08','ses-func02', '02')
msc_08_03_01 = make_timeseries_to_vector('sub-MSC08','ses-func03', '01')
msc_08_03_02 = make_timeseries_to_vector('sub-MSC08','ses-func03', '02')
msc_08_04_01 = make_timeseries_to_vector('sub-MSC08','ses-func04', '01')
msc_08_04_02 = make_timeseries_to_vector('sub-MSC08','ses-func04', '02')
msc_08_05_01 = make_timeseries_to_vector('sub-MSC08','ses-func05', '01')
msc_08_05_02 = make_timeseries_to_vector('sub-MSC08','ses-func05', '02')
msc_08_06_01 = make_timeseries_to_vector('sub-MSC08','ses-func06', '01')
msc_08_06_02 = make_timeseries_to_vector('sub-MSC08','ses-func06', '02')
msc_08_07_01 = make_timeseries_to_vector('sub-MSC08','ses-func07', '01')
msc_08_07_02 = make_timeseries_to_vector('sub-MSC08','ses-func07', '02')
msc_08_08_01 = make_timeseries_to_vector('sub-MSC08','ses-func08', '01')
msc_08_08_02 = make_timeseries_to_vector('sub-MSC08','ses-func08', '02')
msc_08_09_01 = make_timeseries_to_vector('sub-MSC08','ses-func09', '01')
msc_08_09_02 = make_timeseries_to_vector('sub-MSC08','ses-func09', '02')
msc_08_10_01 = make_timeseries_to_vector('sub-MSC08','ses-func10', '01')
msc_08_10_02 = make_timeseries_to_vector('sub-MSC08','ses-func10', '02')
#concatenate them all
msc_08_data = pd.concat([msc_08_01_01, msc_08_01_02, msc_08_02_01, msc_08_02_02, 
                         msc_08_03_01, msc_08_03_02, msc_08_04_01, msc_08_04_02,
                         msc_08_05_01, msc_08_05_02, msc_08_06_01, msc_08_06_02,
                         msc_08_07_01, msc_08_07_02, msc_08_08_01, msc_08_08_02,
                         msc_08_09_01, msc_08_09_02, msc_08_10_01, msc_08_10_02])
#resent the index
msc_08_data = msc_08_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_08_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#test out function on individual datasets for subject 10
msc_10_01_01 = make_timeseries_to_vector('sub-MSC10','ses-func01', '01')
msc_10_01_02 = make_timeseries_to_vector('sub-MSC10','ses-func01', '02')
msc_10_02_01 = make_timeseries_to_vector('sub-MSC10','ses-func02', '01')
msc_10_02_02 = make_timeseries_to_vector('sub-MSC10','ses-func02', '02')
msc_10_03_01 = make_timeseries_to_vector('sub-MSC10','ses-func03', '01')
msc_10_03_02 = make_timeseries_to_vector('sub-MSC10','ses-func03', '02')
msc_10_04_01 = make_timeseries_to_vector('sub-MSC10','ses-func04', '01')
msc_10_04_02 = make_timeseries_to_vector('sub-MSC10','ses-func04', '02')
msc_10_05_01 = make_timeseries_to_vector('sub-MSC10','ses-func05', '01')
msc_10_05_02 = make_timeseries_to_vector('sub-MSC10','ses-func05', '02')
#msc_10_06_01 = make_timeseries_to_vector('sub-MSC10','ses-func06', '01') #looks like this subject does not have these runs
#msc_10_06_02 = make_timeseries_to_vector('sub-MSC10','ses-func06', '02') #looks like this subject does not have these runs
msc_10_07_01 = make_timeseries_to_vector('sub-MSC10','ses-func07', '01')
msc_10_07_02 = make_timeseries_to_vector('sub-MSC10','ses-func07', '02')
msc_10_08_01 = make_timeseries_to_vector('sub-MSC10','ses-func08', '01')
msc_10_08_02 = make_timeseries_to_vector('sub-MSC10','ses-func08', '02')
msc_10_09_01 = make_timeseries_to_vector('sub-MSC10','ses-func09', '01')
msc_10_09_02 = make_timeseries_to_vector('sub-MSC10','ses-func09', '02')
msc_10_10_01 = make_timeseries_to_vector('sub-MSC10','ses-func10', '01')
msc_10_10_02 = make_timeseries_to_vector('sub-MSC10','ses-func10', '02')
#concatenate them all
msc_10_data = pd.concat([msc_10_01_01, msc_10_01_02, msc_10_02_01, msc_10_02_02, 
                         msc_10_03_01, msc_10_03_02, msc_10_04_01, msc_10_04_02,
                         msc_10_05_01, msc_10_05_02, 
                         msc_10_07_01, msc_10_07_02, msc_10_08_01, msc_10_08_02,
                         msc_10_09_01, msc_10_09_02, msc_10_10_01, msc_10_10_02])
#resent the index
msc_10_data = msc_10_data.reset_index()
#rename the old index column as the numeric representation of each movement
msc_10_data.rename(columns={'index':'Block (numeric)'}, inplace=True)

###############################################################################
#Now let's concatenate all of the subjects into one dataframe

msc_data =  pd.concat([msc_01_data, msc_02_data, msc_03_data, msc_04_data, msc_05_data,
                       msc_06_data, msc_07_data, msc_08_data, msc_09_data, msc_10_data])
                       

###############################################################################

'''Now we are going to try some analysis stuff.
First, to do some data frame manipulation and test train splits'''
#Make test train split
x = pd.DataFrame(msc_data.iloc[:,5:])
y = pd.DataFrame(msc_data['Block (numeric)'])
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.2)


#Let's start with a support vector classifier
from sklearn.svm import SVC

#Create a svm Classifier
clf_svm = sklearn.svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf_svm.fit(train_X, train_y.values.ravel())
#Predict the response for test dataset
y_pred = clf_svm.predict(test_X)

#Let's evaluate performance 
print(clf_svm.score(train_X, train_y))
print(clf_svm.score(test_X, test_y))
from sklearn.metrics import classification_report
print(classification_report(test_y, y_pred))
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(test_y, y_pred)
print(cm_svm)


#Let's try logistic regression
#fit the model
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(train_X, train_y.values.ravel())
pred_y = clf_lr.predict(test_X)

#How does it perform?
print(clf_lr.score(train_X, train_y))
print(clf_lr.score(test_X, test_y))
print(classification_report(test_y, pred_y))
cm_lr = confusion_matrix(test_y, pred_y)
print(cm_lr)


#Let's try a random forest classifier
#fit the model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
forest.fit(train_X, train_y.values.ravel())
pred_y = forest.predict(test_X)

#How does it perform?
print(forest.score(train_X, train_y))
print(forest.score(test_X, test_y))
print(classification_report(test_y, pred_y))
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(test_y, pred_y)
print(cm_rf)






















'''This stuff is for trying to load in all the subjects, still a WIP'''

#now let's load in all the data     
for subject in Subject_List:
    for session in Session_list:
        for run in Run_list:
          msc_motor_full = pd.concat([make_timeseries_to_vector(subject, session, run)])



for session in Session_list:
    for run in Run_list:
        msc_motor_full =make_timeseries_to_vector('sub-MSC01', session, run)
        


            
          
msc_motor_full_list = []  
for subject in Subject_List:
    for session in Session_list:
        for run in Run_list:
            msc_motor_full_list.append(make_timeseries_to_vector(subject, session, run))
msc_motor_full = pd.concat(msc_motor_full_list)

msc01_list = []  
for session in Session_list:
    for run in Run_list:
            msc_motor_full_list.append(make_timeseries_to_vector('sub-MSC01', session, run))
msc01_full = pd.concat(msc_motor_full_list)

duplicates = msc01_full.duplicated()

msc01_full['duplicates'] = duplicates
msc01_full = msc01_full[ ['duplicates'] + [ col for col in msc01_full.columns if col != 'duplicates' ] ]
