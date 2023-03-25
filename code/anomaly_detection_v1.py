# Imports
import pandas as pd
import copy
import numpy as np
import ruptures as rpt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
from statistics import mean

def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Data from May 2022 to September 2022
def create_df(data_folder):
    ''' Returns a single df with a column 
    identifying the month and year of data '''
    
    # create an empty pandas data frame
    df = pd.DataFrame()

    # iterate over all files within "Data" folder
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            fname = file.split('_')
            month, year = fname[0], fname[1] 
            df_file = pd.read_csv(os.path.join(data_folder, file))
            df_file['Month'] = month
            df_file['Year'] = year
            df = pd.concat([df , df_file], axis=0)
            
    return df

def clean_time_df(df):
    '''
    Returns a df with new columns for Date, Month,
    Year
    '''
    # Split on "." and index into 0-th element
    df['Time'] = df['Time'].apply(lambda x: x.split(".")[0])
    # Convert to datetime object
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].apply(lambda x: x.day)
    df['Month'] = df['Time'].apply(lambda x: x.month)
    df['Year'] = df['Time'].apply(lambda x: x.year)

    return df

def find_min_segment(df, device_id):
    '''
    Returns minimum segment size (min number of speed tests) for a device
    for 1 week ()
    '''
    # keep ookla & ndt7 tests only
    df_ndt7_ookla = df.loc[(df['Tool']=='ndt7' ) | (df['Tool']=='ookla' )]
    # count tests by device, month, date, protocol 
    df_by_test = df_ndt7_ookla.groupby(['ID', 'Month', 'Date', 'Tool'])['Year'].count().reset_index()
    df_by_test = df_by_test.rename(columns={'Year':'Number of tests'})
    
    # Find minimum number of tests per device, across all dates & months
    df = df_by_test.groupby(['ID'])['Number of tests'].min() # series object
    df_frame = df.to_frame().reset_index()
    df_filter_deviceid = df_frame.loc[df_frame['ID']==device_id]
    
    # Minimum segment length = minimum num of tests x 7 days of week
    if len(df_filter_deviceid)>0:
        min_seg = df_filter_deviceid['Number of tests'][0] * 7
    else:
        min_seg = 7
    return min_seg


def cp_by_device(device_id, df_direction, str_direction, test, cp_algo, cost_fn, baseline):
    
    ''' Returns predicted change points for a given device 
    for a given test protocol and a given direction if df length > minimum segment size, else 
    returns False '''
    
    # Filter by device, test
    df_dir_id_test = df_direction.loc[(df_direction['ID']==device_id) & (df_direction['Tool']==test)]
    
    # 14 is arbotrary threshold to ensure that the number of elements 
    # in speed_list are sufficiently large for PELT to work. PELT requires
    # atleast 2 elements 
    if len(df_dir_id_test) > 14:
        # sort by datetime  
        df_dir_id_test = df_dir_id_test.sort_values(by = 'Time')
        # Predict change point indexes
        speed_list = df_dir_id_test['Speed'].to_numpy()
        time_list = df_dir_id_test['Time'].to_numpy()
        if cp_algo == 'pelt':
            #print(speed_list)
            algo = rpt.Pelt(model=cost_fn).fit(speed_list)
            pred_cp = algo.predict(pen=5) # predicted change point indexes
        elif cp_algo == 'window':
            algo = rpt.Window(model=cost_fn).fit(speed_list)
            pred_cp = algo.predict(pen=5)

        # Determine minimum segment size
        if baseline:
            minimum_segment_size = 7
        else:
            minimum_segment_size = find_min_segment(df_dir_id_test, device_id)

        # Modify change points based on minimum segment size
        pred_cp_final = copy.deepcopy(pred_cp)
        for idx in range(len(pred_cp)):
            # idx isnot the last index of list pred_cp
            if idx < (len(pred_cp)-1):
                if (pred_cp[idx+1] - pred_cp[idx] ) < minimum_segment_size:
                    pred_cp_final.remove(pred_cp[idx])
            elif idx == (len(pred_cp)-1):
            # idx is last index of list pred_cp OR when there is only 1 element in list pred_cp

                # Only 1 element in list pred_cp

                # if difference bw changepoint at idx and 1st index of speed_lst or
                # if difference bw changepoint at idx and last index of speed_lst
                # are less than minimum segment size then remove the changepoint at idx from list of changepoints

                if (len(pred_cp)-1)==0:
                    if ((pred_cp[idx] - 0) < minimum_segment_size) or ((len(speed_list)-1) - pred_cp[idx]) < minimum_segment_size:
                        pred_cp_final.remove(pred_cp[idx])
                # more than one elements in pred_cp but this is the last element
                elif (len(pred_cp)-1) > 0:
                    # current -previous cp < min_seg_size or if last index of speed_lst - current cp < min_seg_size
                    if ((pred_cp[idx] - pred_cp[idx-1]) < minimum_segment_size) or ((len(speed_list)-1) - pred_cp[idx]) < minimum_segment_size:
                        pred_cp_final.remove(pred_cp[idx])

        return pred_cp_final, speed_list
    else:
        return [], []


def cp_devices(df, device_list, tests, cp_algo, cost_fn, baseline):
    ''' Returns a dictionary of devices which contains a dictionary of 
    speed test protocols run on that device, and each speed test contains
    a dictionary of tests run on upload and download speeds, and 
    each upload/download "key" in the dictionary contains list of changepoints 
    and list of speeds for that device. 
    '''
    
    df_down = df[df['Direction'] == 'download']
    df_up = df[df['Direction'] == 'upload']
    direction = {'Upload': df_up, 'Download': df_down }

    device_num = -1
    device = {}

    for device_id in device_list:
        device_num += 1
        device_test = {}
        for test in tests:
            device_test_dir = {}
            for dir_str, dir_df in direction.items():
                cp_lst, speeds_lst = cp_by_device(device_id, dir_df, dir_str, test, cp_algo, cost_fn, baseline)
                device_test_dir[dir_str] = [cp_lst, speeds_lst] #device_id=1, test=ndt7, dir = upload/download is k -> cp for all months 
            device_test[test] = device_test_dir #device_id=1, test=ndt7/ookla/iperf3(tcp)/iperf3(udp) is k
        device[device_id] = device_test
        #print("Device Number", device_num)
        #print(device[device_id])
        
    return device



def is_invalid_cp(cp, idx_of_cp, cp_list, signal, threshold, invalid_cp):
    '''docstring'''
    if idx_of_cp == 0:
        ss1_start = 0
    else:
        # previous changepoint
        ss1_start = cp_list[idx_of_cp - 1]
    ss1_end = cp
    # Mean of sub-signal before changepoint (from start of signal or start from previous changepoint)
    if len(signal[ss1_start:ss1_end]) > 0:
        mean_ss1 = mean(signal[ss1_start:ss1_end])
    else:
        mean_ss1 = 1
    ss2_start = cp
    if idx_of_cp == len(cp_list) - 1:
        ss2_end = len(signal) - 1
    else:
        # next changepoint
        ss2_end = cp_list[idx_of_cp + 1]
    # Mean of sub-signal after changepoint (upto next changepoint or upto end of signal)
    if len(signal[ss2_start:ss2_end]) > 0:
        mean_ss2 = mean(signal[ss2_start:ss2_end])
    else:
        mean_ss2 = 1
    pcent_diff_mean = (abs(mean_ss2 - mean_ss1) / min(mean_ss1, mean_ss2))*100
    # Identify invalid changepoints
    if pcent_diff_mean < threshold:
        invalid_cp.append(cp)
    
    return invalid_cp



def check_valid_cp(device_list, threshold, devices_all, tests, directions):
    '''docstring'''
    devices_all_copy = copy.deepcopy(devices_all)
    
    device_num = 0
    for device_id in device_list:
        device_num += 1
        #print("Device:", device_num)
        #print(device_id)
        for test in tests:
            for direction in directions:
                dev_test_dir = devices_all_copy[device_id][test][direction]
                cp_list, signal = dev_test_dir[0], dev_test_dir[1]
                invalid_cp = []
                # Iterate through changepoints for combination of a 
                # given device, test, direction
                for idx_of_cp,cp in enumerate(cp_list):
                    invalid_cp_lst = is_invalid_cp(cp, idx_of_cp, cp_list, signal, threshold, invalid_cp)
                # Remove invalid changepoints
                for remove_cp in invalid_cp_lst:
                    if len(devices_all_copy[device_id][test][direction][0]) > 0:
                        devices_all_copy[device_id][test][direction][0].remove(remove_cp)
        #print(devices_all_copy[device_id])
    return devices_all_copy  



def plot_cp(pre_heuristics_data, post_heuristics_data, pdf_name, pdf_folder):
    '''
    Returns dashed lines for changepoints detected from
    pre-heuristics and post-heuristics datasets about
    download and upload speed for each
    speed test protocol, for a given device id.
    '''
    
    # Create the full path for the PDF file
    pdf_file_path = os.path.join(pdf_folder, pdf_name)
    
    # Initialize the pdf file
    pp = PdfPages(pdf_file_path)
    
    device_num = 1
    for device_id, tests_dict in pre_heuristics_data.items():
        #print("Device number ", device_num)
        #print("tests_dict", tests_dict)
        for test, direction_dict in tests_dict.items():
            for direction, cp_speed in direction_dict.items():
                pre_pred_cp = cp_speed[0]
                speed_lst = cp_speed[1]
                post_pred_cp = post_heuristics_data[device_id][test][direction][0]
                if (speed_lst is not None) and (len(pre_pred_cp)>0):
                    fig = plt.figure(figsize=(12, 7))
                    sns.lineplot(speed_lst)
                    for idx, pre_cp_index in enumerate(pre_pred_cp):
                        plt.axvline(pre_cp_index, color='yellow',linestyle='dashed', linewidth=2.0)
                        if (len(post_pred_cp) > 0) and (len(post_pred_cp) > idx):
                            post_cp_index = post_pred_cp[idx]
                            plt.axvline(post_cp_index, color='black',linestyle='solid', linewidth=4.0)
                    plt.xticks(post_pred_cp)
                    plt.suptitle("Device id: "+ str(device_num), y=1.05, fontsize=14)
                    str_pre_pred_cp = str([pre_pred_cp])
                    str_post_pred_cp = str([post_pred_cp])
                    plt.title("Pre heuristic: "+str_pre_pred_cp+" \
                    Post heuristic: "+str_post_pred_cp, fontsize=12)
                    plt.ylabel(direction+" speed")
                    plt.xlabel("Index of samples")
                    plt.legend([test])
                    plt.show()
                    pp.savefig(fig)
                    plt.close()
        #if device_num == 3:
        #    break
        device_num += 1
    
    # Close the pdf file
    pp.close()


def main():


    # Install packages through subprocess within script
    install_package("ruptures")
    install_package("seaborn")
    install_package("matplotlib")
    install_package("pandas")
    install_package("numpy")
    
    # Create dataframe
    print("Executing -- Create Dataframe")
    df_initial = create_df("../Data/")
    df = clean_time_df(df_initial)

    # List of all device IDs
    print("Executing -- Create variables")
    device_list = df['ID'].unique().tolist()
    tests = ['ndt7', 'ookla']
    directions = ['Upload', 'Download']

    # Compute changepoints

    print("PELT Search Algorithm & RBF cost: no heuristics")
    ### Change Points using PELT Search Algorithm & RBF cost fn
    # Heuristics implemented: None
    # minimum segment size of a sub-signal is set to baseline i.e. 7
    baseline = True
    devices_pelt_rbf = cp_devices(df, device_list, tests, 'pelt', 'rbf', baseline)
    plot_cp(devices_pelt_rbf, devices_pelt_rbf, 'pelt_rbf_baseline.pdf', '../Data/')

    # Heuristics implemented: Minimum Segment Size
    # minimum segment size of a sub-signal is set to minimum of number of daily tests run on a given device
    print("PELT Search Algorithm & RBF cost: heuristic -- Minimum Segment Size")
    baseline = False
    devices_pelt_rbf_nobaseline = cp_devices(df, device_list, tests, 'pelt', 'rbf', baseline)
    plot_cp(devices_pelt_rbf, devices_pelt_rbf_nobaseline, 'pelt_rbf_nobaseline.pdf', '../Data/')

    # Heuristics implemented: Minimum Segment Size, Mean Difference Thresholding
    print("PELT Search Algorithm & RBF cost: heuristic -- both at 5% mean thresh")
    baseline = False
    devices_pelt_rbf_nobaseline = cp_devices(df, device_list, tests, 'pelt', 'rbf', baseline)
    # Mean thresholding heuristic at 5%
    after5_thresh_cp_nobaseline = check_valid_cp(device_list, 5, devices_pelt_rbf_nobaseline, tests, directions)
    # Mean thresholding heuristic at 5%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_pelt_rbf_nobaseline, after5_thresh_cp_nobaseline, 'pelt_rbf_nobaseline_thresh5.pdf', '../Data/')
    print("PELT Search Algorithm & RBF cost: heuristic -- both at 10% mean thresh")
    # Mean thresholding heuristic at 10%
    after10_thresh_cp_nobaseline = check_valid_cp(device_list, 10, devices_pelt_rbf_nobaseline, tests, directions)
    # Mean thresholding heuristic at 10%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_pelt_rbf_nobaseline, after10_thresh_cp_nobaseline, 'pelt_rbf_nobaseline_thresh10.pdf', '../Data/')
    # Heuristics implemented: Mean Difference Thresholding
    print("PELT Search Algorithm & RBF cost: heuristic -- Mean Thresholding 10%")
    baseline = True
    pelt_rbf_baseline_meanthresh = cp_devices(df, device_list, tests, 'pelt', 'rbf', baseline)
    after10_thresh_cp_baseline = check_valid_cp(device_list, 10, pelt_rbf_baseline_meanthresh, tests, directions)
    # Mean thresholding heuristic at 10%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(pelt_rbf_baseline_meanthresh, after10_thresh_cp_baseline, 'pelt_rbf_baseline_thresh10.pdf', '../Data/')



    ### Change Points using Window Search Algorithm & RBF cost fn
    # Heuristics implemented: None
    # minimum segment size of a sub-signal is set to baseline i.e. 7
    print("Window Search Algorithm & RBF cost: no heuristics")
    baseline = True
    devices_window_rbf_baseline = cp_devices(df, device_list, tests, 'window', 'rbf', baseline)
    # no mean thresholding heuristic
    # hence should only have black solid lines since both pre and post heuristic datasets (of mean thresholding) are same
    plot_cp(devices_window_rbf_baseline, devices_window_rbf_baseline, 'window_rbf_baseline.pdf', '../Data/')
    # Heuristics implemented: Minimum Segment Size, Mean Difference Thresholding
    print("Window Search Algorithm & RBF cost: heuristic -- both at 5% mean thresh")
    baseline = False
    devices_window_rbf_nobaseline = cp_devices(df, device_list, tests, 'window', 'rbf', baseline)
    # Mean thresholding heuristic at 5%
    after5_thresh_nobaseline_window = check_valid_cp(device_list, 5, devices_window_rbf_nobaseline, tests, directions)
    # Mean thresholding heuristic at 5%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_window_rbf_nobaseline, after5_thresh_nobaseline_window, 'window_rbf_nobaseline_thresh5.pdf', '../Data/')
    # Mean thresholding heuristic at 10%
    print("Window Search Algorithm & RBF cost: heuristic -- both at 10% mean thresh")
    after10_thresh_nobaseline_window = check_valid_cp(device_list, 10, devices_window_rbf_nobaseline, tests, directions)
    # Mean thresholding heuristic at 10%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_window_rbf_nobaseline, after10_thresh_nobaseline_window, 'window_rbf_nobaseline_thresh10.pdf', '../Data/')
    
    
    ### Change Points using PELT Search Algorithm & Rank-based cost fn
    # Heuristics: None
    print("PELT Search Algorithm & Rank cost: no heuristic")
    baseline = True
    device_pelt_rank = cp_devices(df, device_list, tests, 'pelt', 'rank', baseline)
    plot_cp(device_pelt_rank, device_pelt_rank, 'pelt_rank_baseline.pdf', '../Data/')
    # Heuristics implemented: Minimum Segment Size, Mean Difference Thresholding
    print("PELT Search Algorithm & Rank cost: both at 5%")
    baseline = False
    devices_pelt_rank_nobaseline = cp_devices(df, device_list, tests, 'pelt', 'rank', baseline)
    # Mean thresholding heuristic at 5%
    after5_thresh_cp_nobaseline_rank = check_valid_cp(device_list, 5, devices_pelt_rank_nobaseline, tests, directions)
    # Mean thresholding heuristic at 5%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_pelt_rank_nobaseline, after5_thresh_cp_nobaseline_rank, 'pelt_rank_nobaseline_thresh5.pdf', '../Data/')
    # Mean thresholding heuristic at 10%
    print("PELT Search Algorithm & Rank cost: both at 10%")
    after10_thresh_cp_nobaseline_rank = check_valid_cp(device_list, 10, devices_pelt_rank_nobaseline, tests, directions)
    # Mean thresholding heuristic at 10%
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(devices_pelt_rank_nobaseline, after10_thresh_cp_nobaseline_rank, 'pelt_rank_nobaseline_thresh10.pdf', '../Data/')
    # Heuristics implemented: Mean Difference Thresholding
    print("PELT Search Algorithm & Rank cost: heuristic -- mean thresholding")
    baseline = True
    pelt_rank_baseline_meanthresh = cp_devices(df, device_list, tests, 'pelt', 'rank', baseline)
    # Mean thresholding heuristic at 5%
    print("PELT Search Algorithm & Rank cost: heuristic -- mean thresholding at 5%")
    after5_thresh_cp_baseline_rank = check_valid_cp(device_list, 5, pelt_rank_baseline_meanthresh, tests, directions)
    # plot_cp(before_heurstics_data, after_heursitics_data)
    plot_cp(pelt_rank_baseline_meanthresh, after5_thresh_cp_baseline_rank, 'pelt_rank_baseline_thresh5.pdf', '../Data/')



if __name__ == "__main__":
    import sys
    sys.exit(main())