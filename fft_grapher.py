import pandas as pd
import numpy as np
import matplotlib
import sys
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift

ELECTRODE_TO_EXAMINE = 'No2WmV'
SECONDS_BETWEEN_READS = 6
TIME_COL_NAME = 'Ts'

RAW_DATA_DIR = '/mnt/d/Internships/RawDataCSVs/'

def main():
  electrodes = ['No2WmV', 'No2AmV', 'OxWmV', 'OxAmV', 'CoWmV', 'CoAmV']
  loc_round_board_combos = [('Donovan', 1, 19), ('El_Cajon', 2, 19),
      ('Shafter', 3, 19), ('Donovan', 4, 19)]
  specific = True

  if specific:
    graph_fft(RAW_DATA_DIR, 'El_Cajon', 2, 19, 'CoWmV', [0, 0.09, 0, 12000])
  else:
    for combo in loc_round_board_combos:
      for electrode in electrodes:
        location, round_num, board = combo
        graph_fft(RAW_DATA_DIR, location, round_num, board, electrode,
            [0, 0.09, 0, 100000])
        completion_str = f'Done graphing {location} Round {round_num} Board ' + \
            f'{board} {electrode}'
        print(completion_str, file=sys.stderr)

def graph_fft(base_dir, location, round_num, board, electrode, axis):
  deployment_info = location + '_Round' + str(round_num) + '/Board' + str(board)
  reads_file_path = base_dir + deployment_info + '_ReadsFromLog.csv'
  signal_data = get_electrode_data(reads_file_path, electrode)

  fft_data = fft(signal_data)
  power = np.abs(fft_data)

  # only graph positive frequencies
  freq = np.arange(0, int(len(power) / 2) + 1) * (1 / SECONDS_BETWEEN_READS) / \
      len(power)
  power = power[0:int(len(power) / 2) + 1]

  save_file = 'uniform_sampled_fft_graphs/' + deployment_info + '/' + \
      electrode + '.png'
  plt.axis(axis)
  plt.plot(freq, power)
  plt.title(f'{location} Round {round_num} Board {board}')
  plt.xlabel('Frequency (cycles/second)')
  plt.ylabel(f'Magnitude ({electrode})')
  plt.savefig(save_file)
  plt.clf()

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
  
if __name__ == '__main__':
  main()
