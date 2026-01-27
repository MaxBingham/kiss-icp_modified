# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
import csv
import json
from pathlib import Path

import numpy as np
from lfi.apply_fault_model import apply_fault_model


class KITTIOdometryDataset:
    def __init__(self, data_dir, sequence: str, fault_model: str, visibility: float = 0.1, rain_rate: float = 10.0, *_, **__):
        self.sequence_id = str(sequence).zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        self.fault_model = fault_model
        self.visibility = visibility
        self.rain_rate = rain_rate

        #NEW: Setup for data capture 
        results_root = Path(data_dir).parent / "results" / "latest"
        self.save_dir = results_root / f"faulty_scans_{self.fault_model}_{self.visibility}_{self.rain_rate}" / self.sequence_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = [] 
        self.sequence = self.sequence_id


        # Load GT Poses (if available)
        if int(sequence) < 11:
            self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
            self.gt_poses = self.load_poses(self.poses_fn)

        # Add correction for KITTI datasets, can be easilty removed if unwanted
        from kiss_icp.pybind import kiss_icp_pybind

        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
        )

    def __getitem__(self, idx):
        return self.scans(idx)

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx]), np.array([])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        # Load original points
        original_points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        original_points = self.correct_kitti_scan(original_points)
        
        # Apply fault
        if self.fault_model != "None":
            faulty_points = apply_fault_model(original_points.copy(), self.fault_model, self.visibility, self.rain_rate)
        else:
            faulty_points = original_points
        
        # Compute metrics (no saving of point clouds)
        scan_idx = int(Path(scan_file).stem)
        metrics = self.compute_metrics(original_points, faulty_points, scan_idx)
        self.metrics.append(metrics)
        
        # Save metrics summary after the last scan
        if scan_idx == len(self.scan_files) - 1:
            self.save_metrics_summary()
        
        return faulty_points

    def compute_metrics(self, original_points, faulty_points, scan_idx):
        """
        Compute per-scan metrics for fog and rain.
        
        Returns 
        - scan_idx: Scan index (0 to N-1).
        - extinction_coeff: Extinction coefficient (fog: 1/visibility; rain: rain_rate/100; else: 0).
        - points_lost: Number of points lost due to fault (original points - faulty points).
        - max_range: Maximum distance of any point in the faulty scan (meters).
        - survival_curve: List of 100 values (ratios of points retained per 1m distance bin, 0-100m).
        - backscattering_ratio: Backscattering ratio (fog: 0.1/visibility; rain: rain_rate/50; else: 0).
        """
        # Extinction coefficient (fog: inversely related to visibility; rain: based on rain_rate)
        if self.fault_model == "fog":
            extinction_coeff = 1.0 / self.visibility
        elif self.fault_model == "rain":
            extinction_coeff = self.rain_rate / 100.0  # Example: scale rain_rate (adjust formula as needed)
        else:
            extinction_coeff = 0.0
        
        # Points lost
        points_lost = len(original_points) - len(faulty_points)
        
        # Maximum range (farthest point distance)
        max_range = np.max(np.linalg.norm(faulty_points, axis=1)) if len(faulty_points) > 0 else 0.0
        
        # Range survival curve: Bin distances (0-100m, 1m bins) and count retained points
        distances = np.linalg.norm(original_points, axis=1)
        bins = np.arange(0, 101, 1)  # 0-100m
        hist_orig, _ = np.histogram(distances, bins=bins)
        distances_faulty = np.linalg.norm(faulty_points, axis=1)
        hist_faulty, _ = np.histogram(distances_faulty, bins=bins)
        survival_curve = hist_faulty / hist_orig  # Ratio retained per bin
        survival_curve = np.nan_to_num(survival_curve, nan=0.0)
        
        # Backscattering ratio (fog: visibility-based; rain: rain_rate-based; placeholder otherwise)
        if self.fault_model == "fog":
            backscattering_ratio = 0.1 / self.visibility  # Example
        elif self.fault_model == "rain":
            backscattering_ratio = self.rain_rate / 50.0  # Example
        else:
            backscattering_ratio = 0.0
        
        return {
            "scan_idx": scan_idx,
            "extinction_coeff": extinction_coeff,
            "points_lost": points_lost,
            "max_range": max_range,
            "survival_curve": survival_curve.tolist(),  # For CSV/JSON
            "backscattering_ratio": backscattering_ratio,
        }

    def save_metrics_summary(self):
        """Save per-scan metrics and aggregate summary."""
        # Save per-scan to CSV
        csv_path = self.save_dir / f"metrics_{self.sequence}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)
        
        # Aggregate: Mean/std per visibility/rain_rate
        survival_curves = np.array([m["survival_curve"] for m in self.metrics])
        aggregated = {
            "fault_model": self.fault_model,
            "visibility": self.visibility,
            "rain_rate": self.rain_rate,
            "mean_extinction_coeff": np.mean([m["extinction_coeff"] for m in self.metrics]),
            "std_extinction_coeff": np.std([m["extinction_coeff"] for m in self.metrics]),
            "mean_points_lost": np.mean([m["points_lost"] for m in self.metrics]),
            "std_points_lost": np.std([m["points_lost"] for m in self.metrics]),
            "mean_max_range": np.mean([m["max_range"] for m in self.metrics]),
            "std_max_range": np.std([m["max_range"] for m in self.metrics]),
            "mean_backscattering_ratio": np.mean([m["backscattering_ratio"] for m in self.metrics]),
            "std_backscattering_ratio": np.std([m["backscattering_ratio"] for m in self.metrics]),
            
            # Survival curve: Mean across scans
            "mean_survival_curve": np.mean(survival_curves, axis=0).tolist(),
            "std_survival_curve": np.std(survival_curves, axis=0).tolist(),
        }
        
        # Save aggregated to JSON
        json_path = self.save_dir / f"summary_{self.sequence}.json"
        with open(json_path, "w") as f:
            json.dump(aggregated, f, indent=4)

    def load_poses(self, poses_file):
        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        
        # Transform poses from camera frame to lidar frame using calibration
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        poses_lidar = np.einsum("ij,njk->nik", np.linalg.inv(Tr), poses)
        poses_lidar = np.einsum("nik,kj->nij", poses_lidar, Tr)
        return poses_lidar

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = np.loadtxt(os.path.join(self.kitti_sequence_dir, "times.txt")).reshape(-1, 1)
        return timestamps

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict
