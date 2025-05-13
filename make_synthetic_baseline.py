import os
import numpy as np
import glob
import pandas as pd
from os.path import join, basename, splitext
from emage_utils.motion_io import beat_format_load


def noisy_baseline_human_mismatch(gt_dir, data_dir, split_csv, out_dir, num_samples=5):
    """
    Create synthetic data by mismatching poses from different files.
    Generates multiple samples per ground truth file.

    Args:
        gt_dir (str): Path to test set directory
        data_dir (str): Path to full dataset directory
        split_csv (str): Path to train/test split CSV file
        out_dir (str): Path to output directory
        num_samples (int): Number of synthetic samples to generate per GT file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Load train/test split CSV
    split_df = pd.read_csv(split_csv)
    train_ids = split_df[split_df['type'] == 'train']['id'].tolist()

    # Get all training set files
    train_files = []
    for train_id in train_ids:
        train_file = join(data_dir, f"{train_id}.npz")
        if os.path.exists(train_file):
            train_files.append(train_file)

    print(f"Found {len(train_files)} training files")

    if len(train_files) < 1: # Need at least one training file
        print("Not enough training files for mismatching")
        return

    # List all npz files in ground truth (test) directory
    gt_files = glob.glob(join(gt_dir, "*.npz"))

    if len(gt_files) < 1:
        print("No test files found")
        return

    # Create mismatched data using test set structure but train set poses
    for i, file_path in enumerate(gt_files):
        # Load original test file data just once per GT file
        try:
            gt_data_template = beat_format_load(file_path)
            original_length = gt_data_template["poses"].shape[0]
        except Exception as e:
            print(f"Error loading GT file {basename(file_path)}: {e}. Skipping.")
            continue

        # Find training files with sufficient length for this GT file
        suitable_train_files = []
        for train_file in train_files:
            try:
                # Quickly check shape without loading everything if possible
                with np.load(train_file) as train_data_check:
                    if "poses" in train_data_check and train_data_check["poses"].shape[0] >= original_length:
                        suitable_train_files.append(train_file)
            except Exception as e:
                 print(f"Warning: Could not check/load train file {basename(train_file)}: {e}")
                 continue


        if not suitable_train_files:
            print(f"Warning: No suitable training file found for {basename(file_path)} with length {original_length}")
            continue

        # Generate num_samples synthetic files
        for j in range(num_samples):
            # Select a random suitable training file (with replacement is okay)
            selected_train_file = np.random.choice(suitable_train_files)

            try:
                # Load selected training file data
                train_data = beat_format_load(selected_train_file)
                train_poses_data = train_data["poses"]

                # Create a copy of the GT data template for modification
                data_to_save = gt_data_template.copy()

                # Replace poses with training set poses (truncate if needed)
                data_to_save["poses"] = train_poses_data[:original_length]

                # Construct output filename with _sample_ suffix
                base_filename, ext = splitext(basename(file_path))
                out_filename = f"{base_filename}_sample_{j+1}{ext}"
                out_path = join(out_dir, out_filename)

                # Save to output file
                np.savez(out_path, **data_to_save)
                # Optional: print progress less frequently or summarize later
                print(f"Created {out_filename} using poses from {basename(selected_train_file)}")

            except Exception as e:
                 print(f"Error processing sample {j+1} for {basename(file_path)} using {basename(selected_train_file)}: {e}")
                 continue # Skip this sample if error occurs

        print(f"Generated {num_samples} samples for {basename(file_path)}")


    print(f"Finished creating mismatched files in {out_dir}")


def noisy_baseline_gaussian_noise(gt_dir, out_dir, num_samples=5, noise_scale=0.02):
    """
    Creates synthetic data by adding Gaussian noise to poses from ground truth files.
    Generates multiple noisy samples per ground truth file.

    Args:
        gt_dir (str): Path to ground truth directory.
        out_dir (str): Path to output directory.
        num_samples (int): Number of noisy samples to generate per GT file.
        noise_scale (float): Standard deviation of the Gaussian noise.
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_files = glob.glob(join(gt_dir, "*.npz"))

    if not gt_files:
        print(f"No ground truth .npz files found in {gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files in {gt_dir}")

    for file_path in gt_files:
        try:
            gt_data = beat_format_load(file_path)
            original_poses = gt_data["poses"]
        except Exception as e:
            print(f"Error loading GT file {basename(file_path)}: {e}. Skipping.")
            continue

        for j in range(num_samples):
            try:
                data_to_save = gt_data.copy() # Start with a copy of the original loaded data
                noisy_poses = original_poses + np.random.normal(0, noise_scale, original_poses.shape)
                data_to_save["poses"] = noisy_poses.astype(original_poses.dtype) # Ensure same dtype

                base_filename, ext = splitext(basename(file_path))
                out_filename = f"{base_filename}_sample_{j+1}{ext}"
                out_path = join(out_dir, out_filename)

                np.savez(out_path, **data_to_save)
                # print(f"Created {out_filename} with Gaussian noise (scale: {noise_scale})")
            except Exception as e:
                print(f"Error generating noisy sample {j+1} for {basename(file_path)}: {e}")
                continue
        print(f"Generated {num_samples} noisy samples for {basename(file_path)}")

    print(f"Finished creating noisy baseline (Gaussian) files in {out_dir}")


def noisy_baseline_attenuated(gt_dir, out_dir, num_samples=5, attenuation_ratio=0.8):
    """
    Creates synthetic data by attenuating the original motion.
    Each frame's pose is replaced by a weighted sum of the frame pose and the mean pose,
    reducing the overall motion magnitude.

    Args:
        gt_dir (str): Path to ground truth directory.
        out_dir (str): Path to output directory.
        num_samples (int): Number of attenuated samples to generate per GT file.
        attenuation_ratio (float): Weight for the mean pose (in [0, 1]).
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_files = glob.glob(join(gt_dir, "*.npz"))

    if not gt_files:
        print(f"No ground truth .npz files found in {gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files in {gt_dir}")

    for file_path in gt_files:
        try:
            gt_data = beat_format_load(file_path)
            original_poses = gt_data["poses"]
        except Exception as e:
            print(f"Error loading GT file {basename(file_path)}: {e}. Skipping.")
            continue

        # Compute mean pose over all frames.
        mean_pose = np.mean(original_poses, axis=0)

        for j in range(num_samples):
            try:
                data_to_save = gt_data.copy()
                # Attenuate motion: combine each frame with the mean pose.
                attenuated_poses = (1 - attenuation_ratio) * original_poses + attenuation_ratio * mean_pose
                data_to_save["poses"] = attenuated_poses.astype(original_poses.dtype)

                base_filename, ext = splitext(basename(file_path))
                out_filename = f"{base_filename}_sample_{j+1}{ext}"
                out_path = join(out_dir, out_filename)

                np.savez(out_path, **data_to_save)
                print(f"Created {out_filename} with attenuation (ratio: {attenuation_ratio})")
            except Exception as e:
                print(f"Error generating attenuated sample {j+1} for {basename(file_path)}: {e}")
                continue

        print(f"Generated {num_samples} attenuated samples for {basename(file_path)}")

    print(f"Finished creating attenuated baseline files in {out_dir}")


def noisy_baseline_fluctuation(gt_dir, out_dir, num_samples=5, frequency=100, amplitude=0.05):
    """
    Creates synthetic data by adding rapid sinusoidal fluctuations
    to poses from ground truth files.

    Each frame's pose is modified by adding a sine wave fluctuation,
    with a random phase shift for each generated sample.

    Args:
        gt_dir (str): Path to ground truth directory.
        out_dir (str): Path to output directory.
        num_samples (int): Number of fluctuation samples to generate per GT file.
        frequency (float): Frequency of the sine fluctuation.
        amplitude (float): Amplitude (scale) of the sine fluctuation.
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_files = glob.glob(join(gt_dir, "*.npz"))

    if not gt_files:
        print(f"No ground truth .npz files found in {gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files in {gt_dir}")

    for file_path in gt_files:
        try:
            gt_data = beat_format_load(file_path)
            original_poses = gt_data["poses"]
        except Exception as e:
            print(f"Error loading GT file {basename(file_path)}: {e}. Skipping.")
            continue

        num_frames = original_poses.shape[0]

        for j in range(num_samples):
            try:
                # Generate a random phase shift for variability between samples.
                phase = np.random.uniform(0, 2 * np.pi)
                # Create a sinusoidal fluctuation: shape becomes (T, 1) for broadcasting.
                t = np.arange(num_frames)
                fluctuation = amplitude * np.sin(2 * np.pi * frequency * t / num_frames + phase)
                fluctuation = fluctuation[:, np.newaxis]
                # Add the fluctuation to the original poses.
                fluctuated_poses = original_poses + fluctuation
                fluctuated_poses[:, :3] = original_poses[:, :3]  # Keep the root rotation unchanged
                data_to_save = gt_data.copy()
                data_to_save["poses"] = fluctuated_poses.astype(original_poses.dtype)

                base_filename, ext = splitext(basename(file_path))
                out_filename = f"{base_filename}_sample_{j+1}{ext}"
                out_path = join(out_dir, out_filename)

                np.savez(out_path, **data_to_save)
                print(f"Created {out_filename} with fluctuation (frequency: {frequency}, amplitude: {amplitude})")
            except Exception as e:
                print(f"Error generating fluctuation sample {j+1} for {basename(file_path)}: {e}")
                continue

        print(f"Generated {num_samples} fluctuation samples for {basename(file_path)}")

    print(f"Finished creating fluctuation baseline files in {out_dir}")


def noisy_baseline_fixing(gt_dir, out_dir, num_samples=5):
    """
    Creates synthetic data by fixing the poses of random joints (from 55 joints)
    to their first frame values for every sample.

    Args:
        gt_dir (str): Path to ground truth directory.
        out_dir (str): Path to output directory.
        num_samples (int): Number of samples to generate per GT file.
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_files = glob.glob(join(gt_dir, "*.npz"))

    if not gt_files:
        print(f"No ground truth .npz files found in {gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files in {gt_dir}")

    for file_path in gt_files:
        try:
            gt_data = beat_format_load(file_path)
            original_poses = gt_data["poses"]
        except Exception as e:
            print(f"Error loading GT file {basename(file_path)}: {e}. Skipping.")
            continue

        for j in range(num_samples):
            try:
                # Randomly select joint indices to fix (0 to 54)
                fix_joints = np.random.choice(55, size=18, replace=False)
                data_to_save = gt_data.copy()
                poses_fixed = original_poses.copy()
                for fix_joint in fix_joints:
                    start_idx = fix_joint * 3
                    end_idx = (fix_joint + 1) * 3
                    poses_fixed[:, start_idx:end_idx] = poses_fixed[0, start_idx:end_idx]
                data_to_save["poses"] = poses_fixed.astype(original_poses.dtype)

                base_filename, ext = splitext(basename(file_path))
                out_filename = f"{base_filename}_sample_{j+1}{ext}"
                out_path = join(out_dir, out_filename)

                np.savez(out_path, **data_to_save)
                print(f"Created {out_filename} with fixed joint indices {fix_joints.tolist()}")
            except Exception as e:
                print(f"Error generating fixing sample {j+1} for {basename(file_path)}: {e}")
                continue

        print(f"Generated {num_samples} fixing samples for {basename(file_path)}")

    print(f"Finished creating fixing baseline files in {out_dir}")


if __name__ == "__main__":
    # Define base directory
    BEAT_DIR = "/media/yw/work2/PantoMatrix/BEAT2/beat_english_v2.0.0"

    # Define paths
    gt_dir = "./examples/BEAT2/"  # Test set
    train_data_dir = join(BEAT_DIR, "smplxflame_30")
    split_csv = join(BEAT_DIR, "train_test_split.csv")

    num_samples_to_generate = 5 # Define number of samples for all relevant baselines

    # mismatch
    out_dir_mismatch = "./examples/motion_generated/human mismatch/"
    noisy_baseline_human_mismatch(gt_dir, train_data_dir, split_csv, out_dir_mismatch, num_samples=num_samples_to_generate)

    # gaussian noise
    out_dir_gaussian = "./examples/motion_generated/gaussian noise/"
    noisy_baseline_gaussian_noise(gt_dir, out_dir_gaussian, num_samples=num_samples_to_generate)

    # attenuated
    out_dir_attenuated = "./examples/motion_generated/attenuated/"
    noisy_baseline_attenuated(gt_dir, out_dir_attenuated, num_samples=num_samples_to_generate)

    # rapid motion fluctuation
    out_dir_fluctuation = "./examples/motion_generated/fluctuation/"
    noisy_baseline_fluctuation(gt_dir, out_dir_fluctuation, num_samples=num_samples_to_generate)

    # random fixing
    out_dir_fixing = "./examples/motion_generated/fixing/"
    noisy_baseline_fixing(gt_dir, out_dir_fixing, num_samples=num_samples_to_generate)
