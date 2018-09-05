import pandas as pd
import numpy as np
import matplotlib
import sys
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal

SECONDS_BETWEEN_READS = 6
TIME_COL_NAME = 'Ts'
RAW_DATA_DIR = '/mnt/d/Internships/RawDataCSVs/'

def graph_fft(base_dir, location, round_num, board, electrode, axis, fig_num):
  deployment_info = location + '_Round' + str(round_num) + '/Board' + str(board)
  reads_file_path = base_dir + deployment_info + '_ReadsFromLog.csv'
  signal_data = get_electrode_data(reads_file_path, electrode)

  fft_data = fft(signal_data, n=2048)
  power = np.abs(fft_data)

  # only graph positive frequencies
  freq = np.arange(0, int(len(power) / 2) + 1) * (1 / SECONDS_BETWEEN_READS) / \
      len(power)
  power = power[0:int(len(power) / 2) + 1]

  if not axis is None:
    plt.axis(axis)
    
  fig = plt.figure(fig_num)
  plt.plot(freq, power)
  plt.show(fig)
  
  plt.title(f'{location} Round {round_num} Board {board} {electrode}')
  plt.xlabel('Frequency (cycles/second)')
  plt.ylabel(f'Magnitude ({electrode})')
  plt.yscale('log')

def get_electrode_data(reads_file, electrode_col_name):
  reads_df = pd.read_csv(reads_file)
  cleaned_df = remove_warmup_hour(reads_df)
  electrode_data = closest_point_uniform_sampling(cleaned_df,
      electrode_col_name, SECONDS_BETWEEN_READS)
  return electrode_data

def remove_warmup_hour(df):
  return df[360:]

def closest_point_uniform_sampling(df, electrode, interval):
  uniform_samples = []

  time_sorted_df = df.sort_values([TIME_COL_NAME])
  sorted_times = np.asarray(time_sorted_df[TIME_COL_NAME])

  start_time = time_sorted_df.iloc[0][TIME_COL_NAME]
  end_time = time_sorted_df.iloc[len(df) - 1][TIME_COL_NAME]
  times = np.arange(start_time, end_time, interval)

  for time in times:
    closest_point_index = (np.abs(sorted_times - time)).argmin()
    uniform_samples.append(time_sorted_df.iloc[closest_point_index][electrode])

  return uniform_samples

class FigureNumber(object):
  fig_num = 1

def get_fig_num():
  num = FigureNumber.fig_num
  FigureNumber.fig_num += 1
  return num
