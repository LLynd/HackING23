import numpy as np
import pandas as pd 
import json


def ind2label(outputs, filenames):
    files_labels = {}

    # ranges and classes inds
    ind = np.load('class_idx.npy')
    ranges = pd.DataFrame(ind, columns=['begin_ind', 'end_ind', 'classes'])
    # classes names
    f  =  open("label2id_final.json")
    class_names = json.load(f)

    for i, output in outputs:
        ind_max = output[0]  # first index is the best prediction
        for j, begin in enumerate(ranges.begin_ind):
            if ind_max < begin:
                ind = j-1
                break
        files_labels[filenames[i]] = class_names[ind]
    
    return files_labels


def save_to_json(dict_name, output):
    with open('output.json', "w") as f:
        json.dump(dict_name, f)






    


