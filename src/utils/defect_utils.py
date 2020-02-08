import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

class featureset():
    """
    Generate dataframe containing features for classification
    """
    def __init__(self, obj, peak_offset=1, window_offset=0.5):
        self.peak_offset   = peak_offset
        self.window_offset = window_offset
        self.defects      = makeDefectDF(obj,
                                        peak_offset=peak_offset,
                                        window_offset=window_offset)
        self.switches     = makeGenericDF(obj, "switches",
                                          peak_offset=peak_offset,
                                          window_offset=window_offset)
        self.ins_joints   = makeGenericDF(obj, "insulationjoint",
                                          peak_offset=peak_offset,
                                          window_offset=window_offset)

    def makeDefects(self, obj):
        self.defect11     = makeDefectDF(obj, "AXLE_11")
        self.defect12     = makeDefectDF(obj, "AXLE_12")
        self.defect41     = makeDefectDF(obj, "AXLE_41")
        self.defect42     = makeDefectDF(obj, "AXLE_42")
        self.defects      = pd.concat([self.defect11,
                                       self.defect12,
                                       self.defect41,
                                       self.defect42])

        return self.defects

    def makeSwitches(self, obj):
        """
        DEPRECATED
        """
        self.switches11   = makeSwitchesDF(obj, "AXLE_11")
        self.switches12   = makeSwitchesDF(obj, "AXLE_12")
        self.switches41   = makeSwitchesDF(obj, "AXLE_41")
        self.switches42   = makeSwitchesDF(obj, "AXLE_42")
        self.switches     = pd.concat([self.switches11,
                                       self.switches12,
                                       self.switches41,
                                       self.switches42])
        return self.switches

    def makeInsJoints(self, obj):
        """
        DEPRECATED
        """
        self.ins_joints11 = makeInsulationJointsDF(obj, "AXLE_11")
        self.ins_joints12 = makeInsulationJointsDF(obj, "AXLE_12")
        self.ins_joints41 = makeInsulationJointsDF(obj, "AXLE_41")
        self.ins_joints42 = makeInsulationJointsDF(obj, "AXLE_42")
        self.ins_joints   = pd.concat([self.ins_joints11,
                                       self.ins_joints12,
                                       self.ins_joints41,
                                       self.ins_joints42])
        return self.ins_joints

def findIndex(timestamps, start, end):
    """
    Given starting and ending time timestamps it returns the indexes
    of the closest timestamps in the first arg
    params:
        timestamps: timestamps array to search within
        start, end: timestamps to be within start and end
    """
    # Finds all indexes which satisfy the condition
    # nonzero gets rid of the non-matching conditions
    indexes = np.nonzero((timestamps >= start) & ( timestamps < end))[0]

    return indexes

def findVehicleSpeed(time, obj):
    """
    Gets the vehicle speed closest to the specified time.
    params:
        time: time at which to get the vehicle speed
        speed_df: needs to be obj.MEAS_DYN.VEHICLE_MOVEMENT_1HZ
    """
    speed_df = obj.MEAS_DYN.VEHICLE_MOVEMENT_1HZ
    speed_times = speed_df['DFZ01.POS.VEHICLE_MOVEMENT_1HZ.timestamp'].values
    speed_values = speed_df['DFZ01.POS.VEHICLE_MOVEMENT_1HZ.SPEED.data'].values

    # Minus 1 since using > and we want value before
    bef = np.nonzero(speed_times > time)[0][0] - 1
    aft = bef + 1

    # Finds the closest timestamp
    idx = np.argmin([abs(speed_times[bef] - time), abs(speed_times[aft] - time)])
    closest = bef + idx # plus 0 for bef, plus 1 for after

    speed = speed_values[closest]

    return speed

def getPeakWindow(von, bis, find_peak_offset, window_offset, acc_time, a):
    """
    First finds the highest peak within a peak finding window.
    Then this highest peak is centered by defining a window offset.
    Then we get the start and end index of this window
    These indexes are then used to index the timestamps and acceleration for the axle
    params:
        von, bis: the start and end of a defect
        find_peak_offset, window_offset:
            the offset of which to search for peak, and the size of the actual
            defect window
        acc_time, a:
            all the accelerationn times and corresponding acceleratoins
    OBS:
        use of np.argmax() since find_peaks() does not work consistently if duplicate heights.
    alternative:
        to findIndexes
        acc_window = a_df[(aaa[time_label] >= von - find_peak_offset) &
                          (aaa[time_label] < bis + find_peak_offset)]
        but current method is faster
    """

    # Accounting for shift between von and bis
    if von > bis:
        tmp = von
        von = bis
        bis = tmp

    # Find all indexes contained within the peak searching window
    indexes = findIndex(acc_time,
                         von - find_peak_offset,
                         bis + find_peak_offset)

    # Get highest peak
    peak_idx = np.argmax(a[indexes]) + indexes[0]

    # Center the peak
    start = int(peak_idx - window_offset)
    end   = int(peak_idx + window_offset)
    if (start < 0) or (end > len(acc_time)):
        raise Warning("Out of bounds for peak centering")

    timestamps    = acc_time[start:end]
    accelerations = a[start:end]
    return timestamps, accelerations

def getSeverity(severity):
    """
    Converts the recorded severity into integer codes
    """
    if 'sehr' in severity:
        return 1
    elif 'hoch' in severity:
        return 2
    elif 'mittel' in severity:
        return 3
    elif 'gering' in severity:
        return 4
    else:
        return -1 # undefined

def getDirection(obj):
    """
    Gets the driving direction of the vehicle for a measurement ride
    """
    direction_label = 'DFZ01.POS.FINAL_POSITION.POSITION.data.direction'
    direction = np.unique(obj.MEAS_DYN.POS_FINAL_POSITION[[direction_label]])

    if len(direction) == 1:
         direction = direction[0]
    else:
        raise Warning("Driving direction not unique")
    return direction

def getSwitchComponent(obj):
    """
    Adds the vehicle direction and returns the switch DataFrame
    """
    component=obj.MEAS_POS.POS_TRACK[obj.MEAS_POS.POS_TRACK['TRACK.data.switchtype']==1]
    df_postrack = component.copy()
    df_postrack['TRACK.data.direction_vehicleref'] = df_postrack['TRACK.data.direction']
    cond_left  = (df_postrack['TRACK.data.direction']=='left') & (df_postrack['DFZ01.POS.FINAL_POSITION.POSITION.data.kilometrage']==1)
    cond_right = (df_postrack['TRACK.data.direction']=='right')& (df_postrack['DFZ01.POS.FINAL_POSITION.POSITION.data.kilometrage']==1)
    df_postrack.loc[cond_left, 'TRACK.data.direction_vehicleref']  = 'right'
    df_postrack.loc[cond_right, 'TRACK.data.direction_vehicleref'] = 'left'
    return df_postrack

def makeDefectDF(obj, axle='all', find_peak_offset=1, window_offset=0.5):
    """
    Makes the defect dataframe containing all relevant features.
    params:
        obj: the gdfz measurement ride
        axle: axle for which to find defect
        peak_offset: time in seconds for which to find the highest peak around a defect
        window_offset: time in seconds for which to center around the highest peak
    """

    if axle == 'all':
        axle = ['AXLE_11', 'AXLE_12', 'AXLE_41', 'AXLE_42']
    else:
        axle = [axle]

    defect_type_names = np.unique(obj.ZMON['ZMON.Abweichung.Objekt_Attribut'])

    d_df      = pd.DataFrame()
    nanosec   = 10**9
    samp_freq = 24000 # per sec
    window_offset = window_offset * samp_freq
    driving_direction = getDirection(obj)

    for ax in axle:
        dict_def_n  = dict.fromkeys(defect_type_names, 0)
        defectToClass   = {defect_type_names[i] : (i + 2)
                           for i in range(len(defect_type_names))}

        time_label     = 'DFZ01.DYN.ACCEL_AXLE_T.timestamp'
        acc_label      = 'DFZ01.DYN.ACCEL_AXLE_T.Z_' + ax + '_T.data'
        acc_time = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[time_label].values
        acc      = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[acc_label].values

        columns = ["timestamps", "accelerations", "window_length(s)",
                   "severity", "vehicle_speed(m/s)", "axle",
                   "campagin_ID", "driving_direction",
                   "defect_type", "defect_length(m)", "line, defect_ID",
                   "class_label"]

        for i, row in tqdm((obj.ZMON).iterrows(), total = len(obj.ZMON), desc="ZMON " + ax):
            von      = row['ZMON.gDFZ.timestamp_von.' + ax[:6]]
            bis      = row['ZMON.gDFZ.timestamp_bis.' + ax[:6]]

            # For detecting point or range defect
            interval  = abs(int(von) - int(bis))/nanosec
            if interval == 0:
                # Point defects
                find_peak_offset = find_peak_offset * nanosec
                vehicle_speed    = findVehicleSpeed(von, obj)
            else:
                ### Just using von and bis
                find_peak_offset = 0
                # Vehicle speed is found at the middle of the interval
                midpoint         = int(( von + bis)/2 )
                vehicle_speed    = findVehicleSpeed(midpoint, obj)

            timestamps, acceleration = getPeakWindow(von, bis,
                                                       find_peak_offset, window_offset,
                                                       acc_time, acc)

            # Each defect type number count
            d_type             = row['ZMON.Abweichung.Objekt_Attribut']
            n                  = dict_def_n[d_type]
            dict_def_n[d_type] = n + 1

            window_length = (timestamps[-1] - timestamps[0]) / nanosec
            severity      = getSeverity(row['ZMON.Abweichung.Dringlichkeit'])
            #print(d_type, row['ZMON.Abweichung.Dringlichkeit'])
            identifier    = (row['ZMON.Abweichung.Linie_Nr'], row['ZMON.Abweichung.ID'])
            defect_length = interval * vehicle_speed

            temp_df = pd.DataFrame([[timestamps, acceleration, window_length,
                                     severity, vehicle_speed, ax,
                                     obj.campaign, driving_direction,
                                     d_type, defect_length, identifier,
                                     defectToClass[d_type]]],
                                   index   = [d_type + "_" + str(n) + "_" + ax],
                                   columns = columns)

            d_df = pd.concat([d_df, temp_df], axis=0)

    return d_df

def makeGenericDF(obj, type, axle='all', peak_offset=1, window_offset=0.5):
    if axle == 'all':
        axle = ['AXLE_11', 'AXLE_12', 'AXLE_41', 'AXLE_42']
    else:
        axle = [axle]

    # Offsets
    nanosec = 10**9
    sampling_freq = 24000
    window_offset = window_offset * 24000
    peak_offset = peak_offset * nanosec

    # datarame
    df = pd.DataFrame()
    driving_direction = getDirection(obj)

    for ax in axle:
        columns = ["timestamps", "accelerations", "window_length(s)",
                   "severity", "vehicle_speed(m/s)", "axle",
                   "campagin_ID", "driving_direction"]

        ### DEFECT ###
        if type == 'defect':
            raise Warning("Not yet implemented for defects")

        ### INSULATION JOINT ###
        elif type == 'insulationjoint':
            COMPONENT  = obj.DfA.DFA_InsulationJoints
            time_label = "DfA.gDFZ.timestamp." + ax[:-1]
            columns.extend(["ID", "class_label"])

        ### SWITCHES ###
        elif type == 'switches':
            COMPONENT = getSwitchComponent(obj)
            time_label = "DFZ01.POS.FINAL_POSITION.timestamp." + ax[:-1]
            columns.extend(["crossingpath", "track_name",
                            "track_direction", "switch_ID", "class_label"])

        # Accelerometer accelerations
        acc_time_label = 'DFZ01.DYN.ACCEL_AXLE_T.timestamp'
        acc_label  = 'DFZ01.DYN.ACCEL_AXLE_T.Z_' + ax + '_T.data'
        acc_time   = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[acc_time_label].values
        acc        = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[acc_label].values

        count = 0
        for i, row in tqdm(COMPONENT.iterrows(), total = len(COMPONENT), desc=type + " " + ax):
            timestamp = row[time_label]

            if np.isnan(timestamp):
                continue

            timestamps, accelerations = getPeakWindow(
                                            timestamp, timestamp,
                                            peak_offset, window_offset,
                                            acc_time, acc)

            window_length = (timestamps[-1] - timestamps[0]) / nanosec
            severity = 5
            vehicle_speed = findVehicleSpeed(timestamp, obj)

            features = [timestamps, accelerations, window_length,
                        severity, vehicle_speed, ax,
                        obj.campaign, driving_direction]

            ### INSULATION JOINT ###
            if type == 'insulationjoint':
                ID            = row["DfA.IPID"]
                class_label   = 0
                features.extend([ID, class_label])

            elif type == 'switches':
                # timestamp is start_time
                # end_time   = row[ax_time_label] + row[end_time_label] - row[timestamp_label]
                switch_id  = row['TRACK.data.gtgid']
                track_name = row['TRACK.data.name']
                track_direction = row['TRACK.data.direction_vehicleref']
                crossingpath = str(row["crossingpath"])
                class_label = 1
                features.extend([crossingpath, track_name,
                                track_direction, switch_id, class_label])

            temp_df = pd.DataFrame([features],
                                   index   = [type + "_" + str(count) + "_" + ax],
                                   columns = columns)

            df = pd.concat([df, temp_df], axis=0)
            count += 1

    return df

def savePickle(campaign_objects, identifier, path="AiyuDocs/pickles/"):
    """
    campaign_objects: list of objects

    """
    defects    = pd.DataFrame()
    ins_joints = pd.DataFrame()
    switches   = pd.DataFrame()

    for o in campaign_objects:
        defects = pd.concat([defects, o.defects])
        ins_joints = pd.concat([ins_joints, o.ins_joints])
        switches = pd.concat([switches, o.switches])

    defects.to_pickle(path + identifier + "_defects_df.pickle")
    switches.to_pickle(path + identifier + "_switches_df.pickle")
    ins_joints.to_pickle(path + identifier + "_ins_joints_df.pickle")

##################
### DEPRECATED ###
##################

def makeSwitchesDF(obj, axle):
    """
    DEPRECATED
    Makes a dataframe of ordinary switches and
    params:
        axle: the desired axle channel to work with
    """
    switches = obj.MEAS_POS.POS_TRACK[obj.MEAS_POS.POS_TRACK['TRACK.data.switchtype']==1]

    # The start time of my switch with respect to axle1:
    ax_time_label = 'DFZ01.POS.FINAL_POSITION.timestamp.' + axle[:-1]
    timestamp_label = 'DFZ01.POS.FINAL_POSITION.timestamp'
    end_time_label = 'DFZ01.POS.FINAL_POSITION.timestamp_end'

    time     = 'DFZ01.DYN.ACCEL_AXLE_T.timestamp'
    acc      = 'DFZ01.DYN.ACCEL_AXLE_T.Z_' + axle + '_T.data'
    acc_time = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[time].values
    a        = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[acc].values

    normal_df = pd.DataFrame()
    switches = obj.MEAS_POS.POS_TRACK[obj.MEAS_POS.POS_TRACK['TRACK.data.switchtype']==1]
    switches_time_label  = "DFZ01.POS.FINAL_POSITION.timestamp." + axle[:-1]

    nanosec = 10**9
    find_peak_offset = 1 * nanosec
    window_offset = 12000

    columns = ["timestamps",
               "accelerations",
               "window_length(s)",
               "severity",
               "vehicle_speed(m/s)",
               "crossingpath",
               "driving_direction",
               "axle",
               "class_label"]

    driving_direction = getDirection(obj)

    count = 0
    for i, row in tqdm(switches.iterrows(), total = len(switches), desc="Switches " + axle):

        start_time = row[ax_time_label]
        end_time   = row[ax_time_label] + row[end_time_label] - row[timestamp_label]

        switches_time = row[switches_time_label]

        if np.isnan(switches_time):
            continue

        timestamps, accelerations = getPeakWindow(switches_time, switches_time,
                                     find_peak_offset, window_offset,
                                     acc_time, a)

        severity = 5
        vehicle_speed = findVehicleSpeed(switches_time, obj)
        actual_window_length = (timestamps[-1] - timestamps[0]) / nanosec
        crossingpath = str(row["crossingpath"])
        class_label = 1

        temp_df = pd.DataFrame([[timestamps,
                                 accelerations,
                                 actual_window_length,
                                 severity,
                                 vehicle_speed,
                                 crossingpath,
                                 driving_direction,
                                 axle,
                                 class_label]],
                            index = ["Switches" + "_" + str(count)],
                            columns = columns)

        normal_df = pd.concat([normal_df, temp_df], axis=0)
        count += 1

    return normal_df

def makeInsulationJointsDF(obj, axle, find_peak_offset=1, window_offset=0.5):
    """
    DEPRECATED
    Makes the defect dataframe containing all relevant features.
    params:
        axle: axle for which to find defect
        peak_height: this height determines the peak classification
    """
    time     = 'DFZ01.DYN.ACCEL_AXLE_T.timestamp'
    acc      = 'DFZ01.DYN.ACCEL_AXLE_T.Z_' + axle + '_T.data'
    acc_time = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[time].values
    a        = obj.MEAS_DYN.DFZ01_DYN_ACCEL_AXLE_T[acc].values

    normal_df = pd.DataFrame()
    dfa       = obj.DfA.DFA_InsulationJoints
    insulation_time_label  = "DfA.gDFZ.timestamp." + axle[:-1]

    nanosec = 10**9
    sampling_freq = 24000
    window_offset = window_offset * 24000
    find_peak_offset = find_peak_offset * nanosec

    columns = ["timestamps",
               "accelerations",
               "window_length(s)",
               "severity",
               "vehicle_speed(m/s)",
               "ID",
               "axle",
               "class_label"]

    driving_direction = getDirection(obj)

    count = 0
    for i, row in tqdm(dfa.iterrows(), total = len(dfa), desc="Insulation Joints " + axle):
        insulation_time = row[insulation_time_label]

        timestamps, accelerations = getPeakWindow(insulation_time, insulation_time,
                                     find_peak_offset, window_offset,
                                     acc_time, a)

        actual_window_length = (timestamps[-1] - timestamps[0]) / nanosec
        severity = 5
        vehicle_speed = findVehicleSpeed(insulation_time, obj)
        ID            = row["DfA.IPID"]
        class_label   = 0

        temp_df = pd.DataFrame([[timestamps,
                                 accelerations,
                                 actual_window_length,
                                 severity,
                                 vehicle_speed,
                                 ID,
                                 driving_direction,
                                 axle,
                                 class_label]],
                            index = ["InsulationJoint" + "_" + str(count)],
                            columns = columns)

        normal_df = pd.concat([normal_df, temp_df], axis=0)
        count += 1

    return normal_df
