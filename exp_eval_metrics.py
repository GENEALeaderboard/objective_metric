import glob
import os

import numpy as np
from scipy import linalg
import torch
from emage_evaltools.metric import FGD, BC, L1div
import emage_utils.rotation_conversions as rc
from emage_utils.motion_io import beat_format_load
from emage_utils.motion_rep_transfer import get_motion_rep_numpy
from tqdm import tqdm

device = torch.device("cuda")

def evaluation_emage(joint_mask, gt_list, pred_list, fgd_evaluator, bc_evaluator, l1_evaluator, device):
    fgd_evaluator.reset()
    bc_evaluator.reset()
    l1_evaluator.reset()

    for test_file in tqdm(gt_list, desc="Evaluation"):
        # only load selective joints
        pred_file = [item for item in pred_list if item["video_id"] == test_file["video_id"]][0]
        if not pred_file:
            print(f"Missing prediction for {test_file['video_id']}")
            continue
        # print(test_file["motion_path"], pred_file["motion_path"])
        gt_dict = beat_format_load(test_file["motion_path"], joint_mask)
        pred_dict = beat_format_load(pred_file["motion_path"], joint_mask)

        motion_gt = gt_dict["poses"]
        motion_pred = pred_dict["poses"]
        # motion_pred = gt_dict["poses"] + np.random.normal(0, 1, motion_gt.shape)  # only for metric validation

        betas = gt_dict["betas"]

        t = min(motion_gt.shape[0], motion_pred.shape[0])
        motion_gt = motion_gt[:t]
        motion_pred = motion_pred[:t]

        # bc and l1 require position representation
        motion_position_pred = get_motion_rep_numpy(motion_pred, device=device, betas=betas)["position"]  # t*55*3
        motion_position_pred = motion_position_pred.reshape(t, -1)
        # ignore the start and end 2s, this may for beat dataset only
        audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=2 * 16000,
                                             t_end=int((t - 60) / 30 * 16000))
        motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t - 60, pose_fps=30,
                                               without_file=True)
        bc_evaluator.compute(audio_beat, motion_beat, length=t - 120, pose_fps=30)
        # audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=0 * 16000, t_end=int((t-0)/30*16000))
        # motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=0, t_end=t-0, pose_fps=30, without_file=True)
        # bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=30)

        l1_evaluator.compute(motion_position_pred)

        # fgd requires rotation 6d representaiton
        motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
        motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
        motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55 * 6)
        motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55 * 6)
        fgd_evaluator.update(motion_pred.float(), motion_gt.float())

    metrics = {}
    metrics["fgd"] = fgd_evaluator.compute()
    metrics["bc"] = bc_evaluator.avg()
    metrics["div"] = l1_evaluator.avg()
    return metrics


def make_list(npz_path, audio_basepath='./BEAT2/beat_english_v2.0.0/wave16k/'):
    out_list = []
    npz_files = glob.glob(os.path.join(npz_path, "*.npz"))
    for npz_file in npz_files:
        video_id = os.path.splitext(os.path.basename(npz_file))[0].replace('_output', '')
        motion_path = npz_file
        audio_path = os.path.join(audio_basepath, video_id + '.wav')

        out_list.append({
            "video_id": video_id,
            "motion_path": motion_path,
            "audio_path": audio_path,
            "mode": "test"
        })

    return out_list


def compare_video_ids(list1, list2):
    # check lengths first
    if len(list1) != len(list2):
        print("Lists have different lengths.")
        return False

    # sort both lists by video_id so we compare matching entries
    sorted1 = sorted(list1, key=lambda d: d["video_id"])
    sorted2 = sorted(list2, key=lambda d: d["video_id"])

    for d1, d2 in zip(sorted1, sorted2):
        if d1["video_id"] != d2["video_id"]:
            print(f"Mismatch: {d1['video_id']} != {d2['video_id']}")
            return False

    print("All video_id match!")
    return True


def calculate_diversity(activation: np.ndarray, diversity_times: int = 10_000) -> float:
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist


def calculate_activation_statistics(
    activations: np.ndarray,
) -> (np.ndarray, np.ndarray):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def evaluation_a2p(joint_mask, gt_list, pred_list):
    # this function is modified from https://github.com/facebookresearch/audio2photoreal
    gt_static_list = []
    pred_static_list = []
    gt_diff_list = []
    pred_diff_list = []
    var_list = []

    # Load each sample
    for gt_info, pred_info in zip(gt_list, pred_list):
        gt_dict = beat_format_load(gt_info["motion_path"], joint_mask)
        pred_dict = beat_format_load(pred_info["motion_path"], joint_mask)

        # Expected shape: (T, num_joints, 3)
        gt_pose = gt_dict["poses"]
        pred_pose = pred_dict["poses"]

        # Flatten joints so each frame is a static pose vector
        gt_static = gt_pose.reshape(gt_pose.shape[0], -1)
        pred_static = pred_pose.reshape(pred_pose.shape[0], -1)

        gt_static_list.append(gt_static)
        pred_static_list.append(pred_static)

        # Compute frame differences for kinematic analysis
        gt_diff = gt_static[1:] - gt_static[:-1]
        pred_diff = pred_static[1:] - pred_static[:-1]
        gt_diff_list.append(gt_diff)
        pred_diff_list.append(pred_diff)

        # calculate temporal variance
        var_k = np.var(pred_static, axis=0)
        var_list.append(var_k)

    # Concatenate all frames from each sample
    pred_frames = np.concatenate(pred_static_list, axis=0)
    gt_frames = np.concatenate(gt_static_list, axis=0)
    pred_motion_diff = np.concatenate(pred_diff_list, axis=0)
    gt_motion_diff = np.concatenate(gt_diff_list, axis=0)

    # Compute static diversity (var_g) using all predicted frames
    # var_g = calculate_diversity(pred_frames)
    # print("Static diversity (var_g):", var_g.mean())

    # Compute temporal variance
    var_k = np.concatenate(var_list, axis=0)
    var_k = var_k.mean()

    # Compute static FID (fid_g) using all static frames
    pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_frames)
    gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_frames)
    fid_g = calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g)
    # print("Static FID (fid_g):", fid_g)

    # Compute kinematic FID (fid_k) using differences between frames
    pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_motion_diff)
    gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_motion_diff)
    fid_k = calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k)
    # print("Kinematic FID (fid_k):", fid_k)

    metrics = {
        # "var_g": var_g.mean(),
        "var_k": var_k,
        "fid_g": fid_g,
        "fid_k": fid_k,
    }
    return metrics


if __name__ == '__main__':
    # init
    fgd_evaluator = FGD(download_path="./emage_evaltools/")
    bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
    l1div_evaluator = L1div()

    # get npz lists
    gt_list = make_list('./examples/motion_human')
    pred_list = make_list('./examples/motion_generated')
    if not compare_video_ids(gt_list, pred_list):
        exit()

    # evaluation for fgd, bc, div
    metrics_emage = evaluation_emage([True] * 55, gt_list, pred_list, fgd_evaluator, bc_evaluator, l1div_evaluator, device)

    # evaluation for var_k, fid_g, fid_k
    metrics_a2p = evaluation_a2p([True] * 55, gt_list, pred_list)

    metrics = {**metrics_emage, **metrics_a2p}
    print(metrics)
