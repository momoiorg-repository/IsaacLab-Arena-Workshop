# Franka Robot Imitation Learning Workflow

このガイドは、IsaacLab-ArenaとGR00Tを使用して、Frankaロボットでテレオペレーション、アノテーション、Mimicデータ生成、およびファインチューニングを行う手順を詳細に説明します。

## 1. Record Demonstrations (Teleoperation)

テレオペレーションデバイス（キーボードやSpaceMouseなど）を使用して、人間のデモンストレーションを記録します。

```bash
export DATASET_DIR="/workspaces/isaaclab_arena/output"

# Record
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 \
  --num_demos 3 \
  --num_success_steps 2 \
  table_pick_and_place \
  --embodiment franka \
  --teleop_device keyboard
```

### 引数の説明 (Arguments Explanation):
- `LIVESTREAM=2`: ライブストリーミング（WebRTCなど）を有効にするための環境変数です。Docker内やHeadlessモードでの実行時に可視化するために必要です。
- `--device cpu`: シミュレーションを実行する計算デバイス（CPU/GPU）を指定します。
- `--enable_cameras`: **重要**。視覚ポリシー（Visual Policy）の学習に必要なカメラ観測データ（RGB/Depth）を記録します。
- `--dataset_file`: 記録されたデモンストレーション（HDF5ファイル）を保存するパスを指定します。
- `--num_demos`: 記録を停止するまでの成功デモンストレーション数です。指定数に達すると自動的に終了します。
- `--num_success_steps`: タスク成功と見なすために、成功条件を継続して満たす必要があるステップ数です。
- `table_pick_and_place`: ロードするタスク/環境の名前です。
- `--embodiment franka`: 環境で使用するロボット（Franka Emika Panda）を指定します。
- `--teleop_device keyboard`: ロボットを操作するための入力デバイスです。`keyboard`、`spacemouse` などが選択可能です。

---

## 2. Replay Demonstrations (Verification)

記録されたデモンストレーションを再生して、正しく記録されているか検証するためのオプションステップです。

```bash
# Replay
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
    --dataset_file /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 \
    --device cpu \
    table_pick_and_place
```

---

## 3. Annotate Demonstrations (Mimic Preparation)

記録されたデモに「Mimic」アノテーションを追加します。デモンストレーションをサブタスクに分割したり、Mimic生成に必要なシグナルを追加したりします。

```bash
# Annotate
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py  --device cpu \
  --input_file  /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 \
  --output_file /workspaces/isaaclab_arena/output/table_pick_place_cube_annotated.hdf5 \
  --mimic \
  --enable_cameras \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka
```

```bash
# Annotate
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py  --device cpu \
  --input_file  /workspaces/isaaclab_arena/output/test_demo.hdf5 \
  --output_file /workspaces/isaaclab_arena/output/test_demo_annotated.hdf5 \
  --mimic \
  --enable_cameras \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka
```

### 引数の説明:
- `--input_file`: 手順1で記録した生のHDF5ファイルを指定します。
- `--output_file`: アノテーション済みのデータを保存する新しいHDF5ファイルのパスです。
- `--mimic`: MimicGen/GR00Tと互換性のあるトレーニングデータを生成するために必要な「Mimic」環境モードを有効にします。
- `--object dex_cube`: シーン内のターゲットオブジェクトを指定します（記録時と同じものを指定してください）。

---

## 4. Generate Dataset (Mimic Generation)

アノテーションされたデモンストレーションを元に、シミュレーション上で「再生（Playback）」を行い、より大規模なデータセットを生成します。この際、ドメインランダム化や複数の環境設定を用いることでロバスト性を向上させることができます。

```bash
# Generate Dataset
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --input_file /workspaces/isaaclab_arena/output/test_demo_annotated.hdf5 \
  --output_file /workspaces/isaaclab_arena/output/test_demo_dataset_joint.hdf5 \
  --num_envs 5  \
  --generation_num_trials 20 \
  --mimic \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka_joint
```

### 引数の説明:
- `--num_envs 5`: データ生成のために並列実行する環境の数です。数を増やすと生成は速くなりますが、メモリと計算リソースを消費します。
- `--generation_num_trials 20`: 生成したい成功エピソードの総数です。
- `--mimic`: アノテーションに従ってMimic生成プロセスを実行します。
- `--enable_cameras`: **必須**。生成されるデータセットに画像データを保存するために必要です。

---

## 5. Transform to LeRobot Format

IsaacLabのHDF5データセットを、GR00Tのトレーニングに必要なLeRobotデータセット形式に変換します。

```bash
# Transform to Lerobot
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
    --yaml_file isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml
```

### 引数の説明:
- `--yaml_file`: HDF5データをLeRobot形式にマッピングする方法（観測キー、アクションスペース、カメラ名など）を定義した設定ファイルへのパスです。

---

## 6. Train/Fine-tune GR00T

生成されたデータセットを使用してGR00Tモデルのファインチューニングを行います。
**注意**: このステップは通常GPUが必要であり、専用のDockerコンテナ内で実行されることが一般的です。

```bash
# Train Gr00t (Must be run inside Docker)
./docker/run_docker.sh -g "bash isaaclab_arena_gr00t/scripts/train_gr00t_franka.sh"
```

`train_gr00t_franka.sh` 内部で呼び出される `launch_finetune.py` の主な引数は以下の通りです：

- `--base-model-path`: 事前学習済みのGR00Tモデルチェックポイント（例: `nvidia/GR00T-N1.6-3B`）。
- `--dataset-path`: LeRobot形式に変換されたデータセットのパス。
- `--output-dir`: ファインチューニング後のチェックポイントを保存するディレクトリ。
- `--tune-projector`, `--tune-diffusion-model`: モデルのどの部分を学習させるか指定します（ProjectorとDiffusion Head）。
- `--no-tune-llm`, `--no-tune-visual`: メモリ節約と破滅的忘却（Catastrophic Forgetting）防止のため、LLMバックボーンとVision Encoderを凍結（Freeze）します。
- `--global-batch-size`: 全GPUを通じた合計バッチサイズ。
- `--gradient-accumulation-steps`: GPUメモリが限られている場合に、擬似的に大きなバッチサイズを実現するために勾配を蓄積するステップ数です。

---

## 7. Closed loop inference

```bash
# Closed-Loop Inference (Must be run inside Docker)
./docker/run_docker.sh -g -m ~/VLA-Model
# Then inside the container:
python isaaclab_arena/evaluation/policy_runner.py \
    --device cpu \
    --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
    --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
    --policy_device cuda \
    --enable_cameras \
    --num_steps 200 \
    table_pick_and_place \
    --embodiment franka_joint \
    --object dex_cube
```

## 8. Configuration & Environment Variables

- `export DATASET_DIR="/workspaces/isaaclab_arena/output"`: 出力ファイルのベースディレクトリを設定します。
- `LIVESTREAM=2`: シミュレーションをリモートで確認するためのストリーミングサーバー（通常ポート4700-4900番台）を有効にします。
