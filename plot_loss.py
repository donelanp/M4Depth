import argparse
import logging
import os
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

# configure logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def plot_epoch_RMSE(event_dir: str, csv_path: str, out_path: str):
    """
    Plot RMSE vs epoch.

    Args:
        event_dir (str): The directory containing event files.
        csv_path (str): The path to the .csv file containing validation performance data.
        out_path (str): The path to the desired output image.
    """
    if not os.path.exists(event_dir):
        logger.error(f"{event_dir} does not exist")
    else:
        event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if "events.out.tfevents" in f]

        if len(event_files) == 0:
            logger.error(f"{event_dir} does not contain any event files")
        else:
            # grab RMSE value at each step
            steps = []
            values = []

            for ef in event_files:
                ea = event_accumulator.EventAccumulator(ef)
                ea.Reload()

                if "tensors" not in ea.Tags():
                    logger.warning(f"{os.path.basename(ef)} does not contain 'tensors' tag")
                else:
                    if "epoch_RMSE_log" not in ea.Tags()["tensors"]:
                        logger.warning(f"{os.path.basename(ef)} does not contain 'epoch_RMSE_log' tensor")
                    else:
                        logger.debug(f"{os.path.basename(ef)} has {len(ea.Tensors('epoch_RMSE_log')):02d} steps in the 'epoch_RMSE_log' tensor")

                        for te in ea.Tensors("epoch_RMSE_log"):
                            tv = tf.make_ndarray(te.tensor_proto)
                            steps.append(te.step)
                            values.append(float(tv))

            steps, values = zip(*sorted(zip(steps, values)))

            logger.debug(f"steps=[{', '.join([f'{s:02d}' for s in steps])}]")
            logger.debug(f"values=[{', '.join([f'{v:0.3f}' for v in values])}]")

            plt.rcParams["axes.titleweight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["axes.titlesize"] = 14
            plt.rcParams["axes.labelsize"] = 12
            plt.rcParams["xtick.labelsize"] = 10
            plt.rcParams["ytick.labelsize"] = 10

            # extract validation data
            df = pd.read_csv(csv_path)
            rmsel = df["rmsel"]
            epoch = df["ckpt_name"].apply(lambda x: int(x.replace("cp-", "").replace(".ckpt", "")))

            plt.figure(figsize=(8,5))
            plt.plot(epoch, rmsel.to_numpy(), marker="o", linewidth=2, label="Validation data")
            plt.plot(steps, values, marker="o", label="Train data")
            plt.xticks(range(0, max(steps) + 1, 2))
            plt.xlabel("Epoch")
            plt.ylabel("RMSE (log)")
            plt.title("RMSE (log) vs Epoch")
            plt.grid(True)
            plt.legend()
            plt.savefig(out_path)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Process TensorBoard event directory.")
    parser.add_argument("--event-dir", type=str, required=True, help="Path to the directory containing TensorBoard event files.")
    parser.add_argument("--csv", type=str, required=True, help="Path to validation_perfs.csv")
    parser.add_argument("--out", type=str, default="rmse_log_curve.png", help="Output plot filename")
    args = parser.parse_args()

    # plot RMSE vs epoch for each event file contained in the event directory
    plot_epoch_RMSE(args.event_dir, args.csv, args.out)
