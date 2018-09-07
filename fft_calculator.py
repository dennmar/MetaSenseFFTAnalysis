import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft, fftfreq, fftshift
from bisect import bisect_left

SECONDS_BETWEEN_READS = 6
TIME_COL_NAME = 'Ts'
RAW_DATA_DIR = '/mnt/d/Internships/RawDataCSVs/'

def main():
  electrodes = ['No2WmV', 'No2AmV']
  loc_round_board_combos = [
      ('Donovan', 1, 17), ('Donovan', 1, 19), ('Donovan', 1, 21),
      ('El_Cajon', 2, 17), ('El_Cajon', 2, 19), ('El_Cajon', 2, 21),
      ('Shafter', 3, 17), ('Shafter', 3, 19), ('Shafter', 3, 21),
      ('Donovan', 4, 17), ('Donovan', 4, 19), ('Donovan', 4, 21)
  ]
  freq_bounds = [
      (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08),
      (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08),
      (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08),
      (0.06, 0.07), (0.04, 0.05), (0.07, 0.08), (0.07, 0.08), (0.07, 0.08),
      (0.06, 0.07), (0.06, 0.07), (0.05, 0.06), (0.07, 0.08)
  ]
  should_graph = True
  should_calculate_noise = False

  i = 0
  for combo in loc_round_board_combos:
    for electrode in electrodes:
      location, round_num, board = combo
      freq, fft_power, N_fft = perform_fft(RAW_DATA_DIR, location, round_num,
          board, electrode)

      if should_graph:
        graph_fft(freq, fft_power, location, round_num, board, electrode,
            None)
      if should_calculate_noise:
        freq_start, freq_end = freq_bounds[i]
        noise_power = find_noise_power(freq, fft_power, freq_start, freq_end,
            N_fft)
        i += 1
        print(f'{location} Round {round_num} Board {board} {electrode}')
        print(f'Power of white noise: {noise_power} mV^2')
        print()

      completion_str = f'Done with {location} Round {round_num} Board ' + \
          f'{board} {electrode}'
      print(completion_str, file=sys.stderr)

def perform_fft(base_dir, location, round_num, board, electrode):
  deployment_info = location + '_Round' + str(round_num) + '/Board' + str(board)
  reads_file_path = base_dir + deployment_info + '_ReadsFromLog.csv'
  signal_data = get_signal_data(reads_file_path, electrode)

  fft_coeff = fft(signal_data)
  N_fft = len(fft_coeff)
  fft_power = np.power(np.abs(fft_coeff), 2) / np.power(N_fft, 2)

  # only graph positive frequencies
  freq = np.arange(0, int(len(fft_power) / 2) + 1) * \
      (1 / SECONDS_BETWEEN_READS) / len(fft_power)
  fft_power = fft_power[0:int(len(fft_power) / 2) + 1]

  return (freq, fft_power, N_fft)

def graph_fft(freq, fft_power, location, round_num, board, electrode, axis):
  save_file = 'fft_graphs/' + location + '_Round' + str(round_num) + \
      '/Board' + str(board) + '/' + electrode + '.png'

  if not axis is None:
    plt.axis(axis)
 
  plt.plot(freq, fft_power)
  plt.title(f'{location} Round {round_num} Board {board} {electrode}')
  plt.xlabel('Frequency (cycles/second)')
  plt.ylabel(f'Power ({electrode})')
  plt.yscale('log')
  plt.savefig(save_file)
  plt.clf()
  
def get_signal_data(reads_file, electrode_col_name):
  reads_df = pd.read_csv(reads_file)
  cleaned_df = remove_warmup_hour(reads_df)
  electrode_data = closest_point_uniform_sampling(cleaned_df,
      electrode_col_name, SECONDS_BETWEEN_READS)
  return electrode_data

def remove_warmup_hour(df):
  return df[600:]

def closest_point_uniform_sampling(df, electrode, interval):
  uniform_samples = []

  time_sorted_df = df.sort_values([TIME_COL_NAME])
  sorted_times = np.asarray(time_sorted_df[TIME_COL_NAME])

  start_time = time_sorted_df.iloc[0][TIME_COL_NAME]
  end_time = time_sorted_df.iloc[len(df) - 1][TIME_COL_NAME]
  times = np.arange(start_time, end_time, interval)

  for time in times:
    # the number at the insertion point or right before must be the closest time
    closest_time_index = bisect_left(sorted_times, time)
    alt_closest_time_index = closest_time_index - 1

    if np.abs(sorted_times[closest_time_index] - time) < \
        np.abs(sorted_times[alt_closest_time_index] - time):
      closest_point_index = closest_time_index
    else:
      closest_point_index = alt_closest_time_index

    uniform_samples.append(time_sorted_df.iloc[closest_point_index][electrode])

  return uniform_samples

def find_noise_power(freq, fft_power, freq_start, freq_end, N_fft):
  white_noise_inds = [i for i in range(len(freq)) if freq[i] >= freq_start and
    freq[i] <= freq_end]
  wn_stop_ind = white_noise_inds[len(white_noise_inds) - 1]
  if wn_stop_ind < len(white_noise_inds) - 1:
    wn_stop_ind += 1
  
  fft_white_noise = fft_power[white_noise_inds[0]:wn_stop_ind]

  fft_wn_power_sum = 0
  for wn_power in fft_white_noise:
    fft_wn_power_sum += wn_power

  return (float(fft_wn_power_sum) / len(fft_white_noise)) * N_fft

if __name__ == '__main__':
  main()
