import argparse
import pandas as pd
import subprocess
import time
import re


def get_temperatures(gpu_ids):
        gpu_report = subprocess.getoutput("nvidia-smi -q" + " -i " + ','.join(gpu_ids))
        return [m.group(1) for m in re.finditer(r"GPU Current Temp\s+:\s+(\d+)", gpu_report)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='IDs of GPUs to watch.')
    parser.add_argument('--csv_path', type=str, default='/home/chute/synth/gpu_stats.csv',
                        help='Path to output CSV file for GPU statistics.')
    parser.add_argument('--period', type=int, default=5,
                        help='Number of seconds between each data point.')

    args = parser.parse_args()
    gpu_ids = [i for i in args.gpu_ids.split(',')]

    df = pd.DataFrame(columns=('time', 'gpu_id', 'temperature'))
    start = time.time()
    while True:
        # Sample GPU statistics
        time_since_start = "{:.2f}".format(time.time() - start)
        for gpu_id, temperature in zip(gpu_ids, get_temperatures(gpu_ids)):
            df.loc[len(df)] = (time_since_start, gpu_id, temperature)
            print('gpu: {}, time: {} s, temp: {} C'.format(gpu_id, time_since_start, temperature))

        # Write to the CSV
        df.to_csv(args.csv_path, index=False)

        # Nap
        time.sleep(args.period)
