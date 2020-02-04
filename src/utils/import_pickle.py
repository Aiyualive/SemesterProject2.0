from src.FileOperations import FileOperations as fo
from src.ImportFromHDF5 import track_segment_operations as tso

def import_pickle(filename, directory='/Users/Mac/Documents/STUDY/ETH/SBB/omism_gdfz_analysis/2-Data'):
    print('Importing Pickle: ' + filename + ".pickle")
    t = fo.readpickle(name=filename+'.pickle',rootpath=directory,type_name='20190527')

    trackname='TRACK.data.name'
    kilometrage='DFZ01.POS.FINAL_POSITION.POSITION.data.kilometrage'

    POS_TRACK=t.MEAS_POS.POS_TRACK
    #Make start and end ids of switches clear:
    split_nms=POS_TRACK[trackname].apply(lambda x: t.DfA.split_name(x,to_DfA=True,fill_eq_points=True))
    split_nms1=split_nms.str[0] # name from here
    split_nms2=split_nms.str[1] # to here
    s1,s2,cp=tso.sort_name_direction(split_nms1,split_nms2,POS_TRACK[kilometrage])

    # crossing from s1 to s2, with the switch points: cp
    POS_TRACK[['namefrom','nameto','crossingpath']]=tso.pd.DataFrame(list(zip(s1,s2,cp)),
    	columns=['namefrom','nameto','crossingpath'], index=POS_TRACK.index)
    # Select switches
    Switches=POS_TRACK[POS_TRACK['TRACK.data.switchtype']!=0]
    t.MEAS_POS.POS_TRACK['crossingpath'] = POS_TRACK[['crossingpath']]

    return t
