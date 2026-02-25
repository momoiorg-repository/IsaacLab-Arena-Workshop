# Franka Pick-and-Place ワークフロー ハンズオンガイド

> **対象ワークショップ**: NVIDIA学生アンバサダー — PhysicalAI with IsaacSim & GR00T
> **所要時間**: 90分
> **実行環境**: NVIDIA Brev クラウド（L40s GPU × 1）

---

## 目次

1. [全体像の確認](#0-全体像の確認)
2. [環境セットアップ](#1-環境セットアップ)
3. [テレオペレーション（デモ収録）](#2-テレオペレーション-デモ収録)
4. [Mimicによるデータ生成](#3-mimicによるデータ生成)
5. [LeRobotフォーマット変換](#4-lerobotフォーマット変換)
6. [GR00T N1.6 Post-training](#5-gr00t-n16-post-training)
7. [クローズドループ推論・評価](#6-クローズドループ推論評価)

---

## 0. 全体像の確認

このワークショップでは **Franka Panda ロボットアームによるピック＆プレースタスク** を題材に、
Physical AI の開発サイクルをエンドツーエンドで体験します。

```
[人間がテレオペ] → [デモを録画] → [Mimicで大量生成] → [GR00Tでファインチューン] → [シミュレーションで評価]
      ↑                                                                                       ↓
      ←───────────────────── データが増えるほど精度が上がる ──────────────────────────────────
```

### 使用するキーコンポーネント

| コンポーネント | 役割 |
|---|---|
| **NVIDIA Isaac Sim** | 物理ベースロボットシミュレーター |
| **IsaacLab-Arena** | タスク・ロボット・シーンを簡単に組み合わせるフレームワーク |
| **Isaac Lab Mimic** | 少数デモから大量の多様なデータを自動生成 |
| **GR00T N1.6** | NVIDIA製ロボット基盤モデル（3Bパラメータ） |
| **LeRobot** | ロボット学習用データセット標準フォーマット（Hugging Face） |

### アクションスペースの概念（重要）

Franka ワークフローでは **2つのアクションスペース** を使い分けます。

| フェーズ | embodiment | アクション | 目的 |
|---|---|---|---|
| テレオペ・データ生成 | `franka` | IK制御（エンドエフェクタ姿勢） | 人間が直感的に操作できる |
| データ変換・学習 | `franka` | 関節角度記録（8DOF）| IK解済み関節角度を保存 |
| 推論（クローズドループ） | `franka_joint` | 関節角度制御（8DOF） | GR00T出力と一致 |

> ⚠️ データ生成時は必ず `--embodiment franka` を使ってください。
> `franka_joint` を使うとクラッシュします（IK変換が二重になるため）。

---

## 1. 環境セットアップ

### 1-1. NVIDIA Brev でインスタンスを起動

1. [NVIDIA Brev](https://brev.nvidia.com) にアクセスし、ログイン
2. **New Instance** → GPU: `L40s`（48GB VRAM） を選択
3. インスタンスが起動したら、ターミナルを開く

### 1-2. IsaacLab-Arena リポジトリをクローン

```bash
git clone https://github.com/momoiorg-repository/IsaacLab-Arena-Workshop.git /workspaces/isaaclab_arena
cd /workspaces/isaaclab_arena
```

### 1-3. Docker コンテナの起動

IsaacLab-Arena は Docker コンテナ内で動作します。

```bash
# GR00T コンテナを起動（テレオペ・データ生成・学習・推論すべてに使用）
./docker/run_docker.sh -g -b
```

**オプション:**
- `-g` — GR00T N1.6 依存ライブラリを含むイメージを使用
- `-b` — Brev 環境向けエントリポイント（ユーザー/グループの自動セットアップ）

```bash
./docker/run_docker.sh -g -b \
  -d /ephemeral/dataset \
  -m /ephemeral/model
```


### 1-4. 出力ディレクトリの設定

コンテナ内で以下を実行します。

```bash
export DATASET_DIR="/workspaces/isaaclab_arena/output"
export MODELS_DIR="/workspaces/isaaclab_arena/models"
mkdir -p ${DATASET_DIR} ${MODELS_DIR}
```

---

## 2. テレオペレーション (デモ収録)

**目標**: キーボードで Franka アームを操作し、デモンストレーションを 5 件録画する。

### 2-1. キーボード操作の説明

| キー | 動作 |
|---|---|
| `W` / `S` | エンドエフェクタ 前進 / 後退 (X軸) |
| `A` / `D` | エンドエフェクタ 左 / 右 (Y軸) |
| `Q` / `E` | エンドエフェクタ 上昇 / 下降 (Z軸) |
| `K` | グリッパー 開 / 閉 |
| `Enter` | デモを成功として保存 |
| `R` | デモをリセット（やり直し） |

### 2-2. デモの録画

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
  --num_demos 5 \
  --num_success_steps 2 \
  table_pick_and_place \
  --embodiment franka \
  --object dex_cube \
  --teleop_device keyboard
```

**重要なオプション:**
- `--num_demos 5` — 成功デモを 5 件収集（ワークショップでは時間節約のため 5 件）
- `--num_success_steps 2` — 2ステップ連続で成功判定されるとデモを保存
- `LIVESTREAM=2` — WebRTC でビジュアライズ（ポート 4700〜4900）

**コツ:**
- ゆっくり・なめらかに動かす（IK コントローラが追従するため）
- 1回の動作でピック→プレースを完結させる
- 失敗したら `R` でリセット

### 2-3. 録画の確認（任意）

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
  --device cpu \
  --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
  table_pick_and_place \
  --embodiment franka \
  --object dex_cube
```

録画したデモが正しく再生されれば OK です。

---

## 3. Mimicによるデータ生成

Isaac Lab Mimic は **少数のデモからサブタスク境界を学習し、新しいオブジェクト配置で
大量の成功デモを自動生成**します。

```
[5件の人間デモ] → [サブタスクアノテーション] → [Mimic自動生成] → [数百〜数千件のデータ]
```

### 3-1. デモのアノテーション

Mimic が機能するためには、デモに**サブタスク境界**を付ける必要があります。
Franka ピック＆プレースには 1 つのサブタスクがあります:

1. **Reach** — キューブに手を伸ばす（グリッパーがキューブに触れる直前）

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
  --device cpu \
  --input_file  ${DATASET_DIR}/franka_demo.hdf5 \
  --output_file ${DATASET_DIR}/franka_demo_annotated.hdf5 \
  --mimic \
  --enable_cameras \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka
```

画面の CLI プロンプトに従い、各デモの境界を指定してください。

### 3-2. データセット生成（Mimic）

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu --headless \
  --enable_cameras \
  --input_file  ${DATASET_DIR}/franka_demo_annotated.hdf5 \
  --output_file ${DATASET_DIR}/franka_dataset.hdf5 \
  --num_envs 20 \
  --generation_num_trials 100 \
  --mimic \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka
```

**オプション解説:**
- `--num_envs 20` — 20の並列環境で生成を高速化
- `--generation_num_trials 100` — 100パターンを試行（成功したものを保存）

**所要時間**: 約 5〜10 分（GPU の種類によって変動）

### 3-3. 生成データの確認（任意）

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file ${DATASET_DIR}/franka_dataset.hdf5 \
  table_pick_and_place \
  --embodiment franka \
  --object dex_cube
```

キューブの位置が元のデモとは異なるバリエーションが再生されれば成功です。

---

## 4. LeRobotフォーマット変換

GR00T N1.6 の学習には **LeRobot フォーマット**（Hugging Face標準）が必要です。
HDF5 ファイルを Parquet + MP4 形式に変換します。

```bash
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml
```

**変換される内容:**
- 関節角度 (`processed_actions`) → `action` キー（GR00T の学習ターゲット）
- 関節状態 (`joint_pos`) → `observation.state` キー
- 手首カメラ映像 → `observation.images.ego_view`（MP4）
- 左・右カメラ映像 → `observation.images.left_view` / `right_view`（MP4）

変換後のデータは `${DATASET_DIR}/franka_dataset/lerobot/` に保存されます。

---

## 5. GR00T N1.6 Post-training

> **ワークショップ注記**: 学習には数時間かかります。
> このステップはコマンドを実行して確認するだけにとどめ、
> **事前学習済みチェックポイントを講師から提供します**。

### 5-1. コンテナの確認

Step 1-3 で起動した GR00T コンテナ（`-g -b`）をそのまま使います。
コンテナ内で環境変数を設定:

```bash
export DATASET_DIR="/workspaces/isaaclab_arena/output"
export MODELS_DIR="/workspaces/isaaclab_arena/models"
```

### 5-2. GR00T N1.6 ファインチューン（確認用・実行のみ）

```bash
CUDA_VISIBLE_DEVICES=0 python \
  submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path ${DATASET_DIR}/franka_dataset/lerobot \
  --output-dir ${MODELS_DIR} \
  --modality-config-path isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-llm \
  --no-tune-visual \
  --global-batch-size 16 \
  --max-steps 10000 \
  --num-gpus 1 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --dataloader-num-workers 16 \
  --use-wandb
```

**フリーズするモジュール（重要）:**
- `--no-tune-llm` — LLM バックボーンを凍結（カタストロフィックフォーゲットを防ぐ）
- `--no-tune-visual` — ビジュアルバックボーンを凍結

**更新するモジュール:**
- `--tune-projector` — モダリティプロジェクターを更新
- `--tune-diffusion-model` — アクション生成部を更新

**本番の推奨設定:**
- 1× L40s (48GB): `--max-steps 30000`（約 2〜3 時間）
- 8× L40s: `--max-steps 20000`（約 4〜8 時間）

### 5-3. 事前学習済みモデルの配置（ワークショップ）

講師から提供されたモデルを以下のパスに配置します:

```bash
# 例: チェックポイントが checkpoint-2000 の場合
ls ${MODELS_DIR}/checkpoint-2000/
# → config.json  model.safetensors  training_args.bin  ...
```

---

## 6. クローズドループ推論・評価

GR00T の推論では `--embodiment franka_joint`（直接関節位置制御）を使います。
これはテレオペ・データ生成時の `franka`（IK制御）と**異なる**ことに注意してください。

### 6-1. 推論コンフィグの確認

`isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml` を開き、
`model_path` が学習済みチェックポイントを指していることを確認します。

```bash
# 例: 事前学習済みモデルを使う場合
model_path: /models/franka-gr00t-checkpoints-2000

# 例: 自分で学習した場合
model_path: /workspaces/isaaclab_arena/models/checkpoint-2000
```

### 6-2. クローズドループ推論の実行

```bash
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

**オプション解説:**
- `--policy_device cuda` — GR00T 推論を GPU で実行（物理シミュレーションは CPU）
- `--num_steps 200` — `action_chunk_length=16` なので、約 12 ロールアウト
- `--embodiment franka_joint` — 直接関節位置制御（GR00T 出力と一致）

### 6-3. 並列評価（複数環境）

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --device cpu \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
  --policy_device cuda \
  --enable_cameras \
  --num_steps 2000 \
  --num_envs 10 \
  table_pick_and_place \
  --embodiment franka_joint \
  --object dex_cube
```

評価終了後、コンソールに成功率が表示されます:

```
Metrics: {'success_rate': 0.85, 'num_episodes': 20}
```

---

## トラブルシューティング

### WebRTC 画面が表示されない

```bash
# ポートが開いているか確認
ss -tlnp | grep 4700
# Brev のポートフォワーディング設定を確認
```

### `Action dimension mismatch` エラー

データ生成時に `--embodiment franka_joint` を使っていないか確認してください。
データ生成には必ず `--embodiment franka` を使います。

### CUDA out of memory

- `--num_envs` を減らす
- `--global-batch-size` を減らす（学習時）

### 学習が収束しない

- デモ数を増やす（最低 50〜100 件を推奨）
- `--generation_num_trials` を 100 以上に増やす

---

## ワークフロー全体チェックリスト

- [ ] **Step 1**: 環境セットアップ完了（Brev + Docker）
- [ ] **Step 2**: テレオペ録画完了（`franka_demo.hdf5`）
- [ ] **Step 3a**: アノテーション完了（`franka_demo_annotated.hdf5`）
- [ ] **Step 3b**: Mimic データ生成完了（`franka_dataset.hdf5`）
- [ ] **Step 4**: LeRobot 変換完了（`franka_dataset/lerobot/`）
- [ ] **Step 5**: Post-training 実行確認（または事前済みモデルを配置）
- [ ] **Step 6**: クローズドループ推論・評価完了

---

## 参考リソース

- [IsaacLab-Arena GitHub](https://github.com/isaac-sim/IsaacLab-Arena)
- [NVIDIA GR00T N1.6 モデルカード](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac Lab Mimic ドキュメント](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)
- [LeRobot フォーマット仕様](https://huggingface.co/docs/lerobot)
- [NVIDIA Brev](https://brev.nvidia.com)
