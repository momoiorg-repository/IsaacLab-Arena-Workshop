
import argparse
import h5py
import json
import numpy as np
import pandas as pd
import shutil
import subprocess
import time
import torch
import torchvision
import traceback
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import Any

def wait_for_video_completion(video_path: str, max_wait_time: int = 60, check_interval: float = 0.5) -> bool:
    video_path = Path(video_path)
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if not video_path.exists():
            time.sleep(check_interval)
            continue
        try:
            size1 = video_path.stat().st_size
            time.sleep(check_interval)
            size2 = video_path.stat().st_size
            if size1 == size2 and size1 > 0:
                try:
                    with open(video_path, "rb") as f:
                        f.read(1024)
                    return True
                except OSError:
                    pass
        except OSError:
            pass
        time.sleep(check_interval)
    return False

def get_video_metadata(video_path: str) -> dict[str, Any] | None:
    if not wait_for_video_completion(video_path, max_wait_time=60):
        print(f"Timeout waiting for video completion: {video_path}")
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate", "-of", "json", video_path]
    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den
        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.width": stream["width"],
                "video.height": stream["height"],
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    except Exception as e:
        print(f"Error getting metadata for {video_path}: {e}")
        return None

def write_video_job(queue: mp.Queue, error_queue: mp.Queue):
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            video_path, frames, fps = job
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Assuming frames are [T, H, W, C] and uint8
            torchvision.io.write_video(str(video_path), torch.from_numpy(frames), fps, video_codec="h264")
        except Exception as e:
            error_msg = f"Error creating video {video_path}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            error_queue.put(error_msg)

def convert_franka_to_lerobot(hdf5_path: str, output_dir: str, task_name: str, fps: int = 30):
    hdf5_path = Path(hdf5_path)
    # Output directory name based on input filename
    output_dir = Path(output_dir) / f"{hdf5_path.stem}_lerobot"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    lerobot_data_dir = output_dir 
    lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    (lerobot_data_dir / "meta").mkdir(exist_ok=True)

    f = h5py.File(hdf5_path, "r")
    data_grp = f["data"]
    
    queue = mp.Queue(maxsize=10)
    error_queue = mp.Queue()
    workers = []
    for _ in range(4):
        p = mp.Process(target=write_video_job, args=(queue, error_queue))
        p.start()
        workers.append(p)

    total_episodes = len(data_grp.keys())
    total_frames = 0
    episodes_info = []
    video_paths = {}
    
    # Iterate over episodes
    for ep_idx, demo_key in enumerate(tqdm(data_grp.keys())):
        demo = data_grp[demo_key]
        
        # Read data
        # Check structure: defaults to obs/policy/joint_pos or similar
        # Depending on how it was recorded. 
        # Standard Isaac Lab ActionStateRecorder:
        # demo['obs']['policy']['joint_pos']
        # demo['actions']
        
        try:
            # Try to locate state and action
            if "obs" in demo:
                # Based on user feedback: obs keys are ['actions', 'datagen_info', 'eef_pos', 'eef_quat', 'gripper_pos', 'joint_pos', 'joint_vel']
                # There is no 'policy' key.
                if "joint_pos" in demo["obs"]:
                     joint_pos = demo["obs"]["joint_pos"][:]
                elif "policy" in demo["obs"] and "joint_pos" in demo["obs"]["policy"]:
                     joint_pos = demo["obs"]["policy"]["joint_pos"][:]
                else:
                    print(f"Could not find joint_pos in {demo_key}. Obs keys: {list(demo['obs'].keys())}")
                    continue
            else:
                print(f"Could not find obs in {demo_key}")
                continue

            if "actions" in demo:
                actions = demo["actions"][:]
            elif "action" in demo:
                actions = demo["action"][:]
            else:
                 print(f"Could not find actions in {demo_key}")
                 continue

            length = len(actions)
            # Observations might be length + 1
            if len(joint_pos) > length:
                joint_pos = joint_pos[:length]
            
            # Timestamp
            timestamp = np.arange(length, dtype=np.float64) / fps

            # Create DataFrame
            df_data = {
                "observation.state": [row for row in joint_pos],
                "action": [row for row in actions],
                "timestamp": timestamp,
                "episode_index": np.full(length, ep_idx, dtype=int),
                "frame_index": np.arange(length, dtype=int),
                "index": np.arange(total_frames, total_frames + length, dtype=int),
                "task_index": np.zeros(length, dtype=int),
            }
            
            # Add rewards/done
            reward = np.zeros(length, dtype=float)
            reward[-1] = 1.0
            done = np.zeros(length, dtype=bool)
            done[-1] = True
            df_data["next.reward"] = reward
            df_data["next.done"] = done

            df = pd.DataFrame(df_data)
            
            # Save Parquet
            chunk_idx = ep_idx // 1000
            pq_path = lerobot_data_dir / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
            pq_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq_path)

            # Save Video
            frames = None
            cam_key = None
            
            # Search strategy 1: Look in obs group
            if "obs" in demo:
                possible_keys = ["wrist_cam_rgb", "robot_pov_cam", "wrist_cam", "camera_obs"]
                for k in possible_keys:
                    if k in demo["obs"]:
                        cam_key = k
                        frames = demo["obs"][cam_key][:]
                        break
            
            # Search strategy 2: Look in root for camera_obs group
            if frames is None and "camera_obs" in demo:
                cam_grp = demo["camera_obs"]
                # Try specific likely keys
                possible_subkeys = ["wrist_cam_rgb", "robot_pov_cam", "rgb", "image"]
                for k in possible_subkeys:
                    if k in cam_grp:
                        cam_key = k
                        frames = cam_grp[k][:]
                        break
                # If not found, take the first available key
                if frames is None and len(cam_grp.keys()) > 0:
                    cam_key = list(cam_grp.keys())[0]
                    frames = cam_grp[cam_key][:]
            
            if frames is not None:
                if len(frames) > length:
                    frames = frames[:length]
                
                vid_key = "observation.images.ego_view"
                vid_path = lerobot_data_dir / f"videos/chunk-{chunk_idx:03d}/{vid_key}/episode_{ep_idx:06d}.mp4"
                queue.put((vid_path, frames, fps))
                
                if vid_key not in video_paths:
                    video_paths[vid_key] = vid_path
            
            episodes_info.append({
                "episode_index": ep_idx,
                "tasks": [task_name],
                "length": length
            })
            
            total_frames += length

        except Exception as e:
            print(f"Error processing {demo_key}: {e}")
            traceback.print_exc()

    # Finish video workers
    for _ in range(4):
        queue.put(None)
    for w in workers:
        w.join()
        
    f.close()

    # Write Meta
    import json
    # info.json
    info = {
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [9], "names": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]},
            "action": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        }
    }
    # update features with video meta
    if video_paths:
        info["total_videos"] = total_episodes
        for k, v in video_paths.items():
            meta = get_video_metadata(v)
            if meta:
                info["features"][k] = meta
    else:
        # No videos
        print("No video data found in HDF5. Dataset will contain only state/action data.")

    with open(lerobot_data_dir / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)
        
    # tasks.jsonl
    with open(lerobot_data_dir / "meta/tasks.jsonl", "w") as f:
        print(json.dumps({"task_index": 0, "task": task_name}), file=f)
        
    # episodes.jsonl
    with open(lerobot_data_dir / "meta/episodes.jsonl", "w") as f:
        for ep in episodes_info:
            print(json.dumps(ep), file=f)

    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_file", type=str)
    parser.add_argument("--task_name", type=str, default="pick_place")
    args = parser.parse_args()
    
    convert_franka_to_lerobot(args.hdf5_file, Path(args.hdf5_file).parent, args.task_name)
