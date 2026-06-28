# V-DASH システムアーキテクチャ — VLA → ルールベース ハンドオフ（M6 スイッチャ）

対象: V-DASH「VLA（知覚・把持・搬送）＋ ルールベース（精密嵌合）」階層制御の **引き継ぎ
（handoff）機構**。本ドキュメントは「現状の V-DASH 実装でハンドオフがどう動くか」「VLA から
ルールベースへどう切り替わるか」をコードに即して説明する。`docs/milestones/` に M6 スイッチャの
個別レポートが無いため、本ドキュメントがその役割（M6 の設計記録）も兼ねる。

### 関連ドキュメント（既存・本書はこれらを統合する）

| ドキュメント | 内容 |
|---|---|
| `docs/V-DASH_dev_brief_v2.md` | 設計指針（§2.1 入力制約 / §3.3 コントローラ / §3.4 述語 / §3.5 ロガー） |
| `docs/vla_interface.md` | **凍結された VLA 前段の契約**（観測・アクション空間・ハンドオフ境界・言語・データ形式） |
| `docs/milestones/M3.md` | スクリプト専門家＋ルールベース嵌合コントローラ（§2.1 準拠化、力 35→4N） |
| `docs/milestones/M5.md` | E2 ルールベース・クリアランス成功曲線（c=2.0 で L0–L3 98–100%） |
| `docs/milestones/M8.md` | 古典ビジョン把持ベースライン（`vdash_rule_based`） |
| `docs/milestones/M9.md` | VLA デモ収集 + インターフェイス凍結（教師データ生成） |
| `docs/vdash_M1-M3_report_ja.md` | M1〜M3 の実装記録（日本語） |

---

## 0. 一言でいうと

> **VLA がピック→搬送（pick→handoff）を担い、ペグが socket 口元の「ハンドオフ円柱」に
> 静止把持で入った瞬間（§3.4 `handoff` 述語が真）に、その env だけルールベースの嵌合
> コントローラへ *ラッチ* で切り替わる。切り替えはポリシーの内部状態ではなく、シミュレータ
> 状態から計算される述語で決まる。**

VLA はニューラルネットなので「もう終わった？」と内部状態を問い合わせられない。そこで
切り替えの判断を **環境側の観測可能な物理状態**（把持・位置・速度）に外出しした述語に委ねる、
というのがこの設計の核心である。`vla_interface.md`（凍結契約）も明記する:

> *“The VLA episode is considered complete when this predicate first holds. **The M6 switcher uses
> the same predicate** to transfer control to the insertion controller.”*

つまり「学習データの切り出し点」と「ランタイムの切り替え点」が **同一の述語** で定義されている。

---

## 1. 階層制御の全体像

```
                ┌──────────────────────────────────────────────────────────┐
                │                  VDashVLAPolicy  (policy "vdash_vla")      │
                │              isaaclab_arena/policy/vdash_vla_policy.py     │
                │                                                            │
   camera RGB   │   ┌─────────────────┐                                     │
   + robot state│   │  VLA 前段         │  pick_act (N,8) 関節目標            │
   ────────────►│──►│  GR00T N1.6      │──────────────┐                     │
                │   │  (closed-loop)   │              │                     │
                │   └─────────────────┘              ▼                     │
                │                          torch.where(in_insertion,        │
   sim 状態      │   ┌─────────────────┐    ins_act, pick_act)  ──► action  │──► env.step
   ────────────►│──►│ §3.4 handoff 述語 │──► in_insertion (N,) bool ラッチ   │   (N,8 関節)
                │   └─────────────────┘              ▲                     │
                │                                    │                     │
                │   ┌─────────────────┐  ins_ik(N,7) │  IK→関節ブリッジ     │
                │   │ InsertionCtrl    │──────────────┘  (DLS IK)           │
                │   │ (rule-based)     │   ins_act (N,8)                     │
                │   └─────────────────┘                                     │
                └──────────────────────────────────────────────────────────┘
```

- **前段（VLA）**: `Gr00tClosedloopPolicy`。カメラ RGB（手首/左/右の3系統）＋関節状態から、
  8-dof の **関節位置目標** をチャンク（`action_horizon=16`）で出力する。
  評価環境は `franka_joint` 体（8-dof 関節位置アクション）なので、VLA 出力はそのまま適用できる。
- **後段（ルールベース）**: `InsertionController`。力制限プレス＋スパイラルサーチでペグを穴に嵌める。
  出力は 7-vector の **相対 IK** アクション `[dx,dy,dz,ax,ay,az,grip]`。
- **スイッチ**: `§3.4 handoff` 述語が env ごとに真になったら、その env を後段へ **ラッチ**。

### 3つの V-DASH ポリシー（前段の違い・後段は共有）

| ポリシー名 | 前段（ピック・搬送） | 後段（嵌合） | 切り替えトリガ | 用途・関連 |
|---|---|---|---|---|
| `vdash_scripted` | スクリプト専門家（特権・真値参照） | `InsertionController` | ピックの `finished` フラグ | 教師データ生成（M9）・成功曲線（M5） |
| `vdash_rule_based` | スクリプト専門家＋古典ビジョン把持 | `InsertionController` | ピックの `finished` フラグ | ルールベース単体ベースライン（M8） |
| **`vdash_vla`** | **GR00T VLA（学習）** | `InsertionController` | **§3.4 `handoff` 述語** | **VLA 評価（M6・本書の主題）** |

3つとも後段は同一の `InsertionController` を共有する。違いは前段と、後段へラッチする
**トリガの取り方** だけである（後述 §4.2）。

---

## 2. ハンドオフ述語（§3.4 `handoff`）— 切り替えの判定条件

`isaaclab_arena_environments/mdp/vdash_predicates.py` の `handoff()`。
シーンテンソルだけで決まる純関数（torch のみ、Isaac Sim 非依存 → ユニットテスト可能）。
返り値は `(num_envs,)` の bool。

```
handoff  ≡  grasped  ∧  (ペグ先端が「ハンドオフ円柱」内)  ∧  (ペグ速度 < speed_max)
```

3条件すべてが同時に真の env だけ `True`。

1. **`grasped`**: グリッパ開度 `width ∈ (width_min, width_max)` ∧ ペグ↔指のフィルタ接触力 `> grasp_force`
2. **ハンドオフ円柱の内側**: ペグ先端の横ずれ `lateral < r_h` ∧ 口元からの高さ `h_low < height < h_high`
3. **ほぼ静止**: ペグ重心速度 `< speed_max`

### 閾値（`configs/vdash/task.yaml` に外出し）

| パラメータ | 値 | 意味 |
|---|---|---|
| `grasped.width_min / width_max` | 0.005 / 0.025 m | 把持と判定するグリッパ開度の窓 |
| `grasped.grasp_force` | 1.0 N | ペグ↔指のフィルタ接触力しきい |
| `handoff.r_h` | 0.015 m | ハンドオフ円柱の半径 |
| `handoff.h_low` | 0.020 m | 円柱下端（口元からの高さ） |
| `handoff.h_high` | 0.060 m | 円柱上端 |
| `handoff.speed_max` | 0.05 m/s | 静止判定の速度上限 |
| `geometry.mouth_height` | 0.030 m | socket root から口元までの高さ |

> なぜ「円柱＋静止」なのか: 口元の真上 20〜60mm に静止して入った状態を要求することで、
> 後段の `InsertionController` が必ず **SETTLE フェーズの入口**（口元少し上で減速・整列）から
> 始められる。接近の運動量を残したまま切り替えると口縁衝突の力スパイクが出る。M3 でも、
> SETTLE のドウェル高さ（35mm）は §3.4 円柱（20〜60mm）の **内側** に置き、`handoff` が
> 50/50 確実に発火するよう整合させている。

---

## 3. ハンドオフの2つの局面

「ハンドオフ」という語は V-DASH では2つの文脈で使われる。混同しないこと。

### 3.1 データ記録時のハンドオフ（学習データの切り出し点・M9）

`scripts/vdash/record_vla_demos.py`。教師データの収集では（詳細は `docs/milestones/M9.md`）：

- 特権スクリプト専門家（`vdash_scripted`）がアームを駆動（§2.1 はデモ生成での専門家使用を許可）。
- **reset から §3.4 `handoff` 述語が初めて真になるまで** を1エピソードとして記録し、そこで切って
  成功デモとしてエクスポートする（`--until handoff`）。
- **嵌合（insertion）は記録しない** — それは後段ルールベースの仕事であり、VLA に学習させない。

つまり VLA は「ピック→口元上で静止把持」までだけを模倣学習する。これがランタイムの切り替え点と
一致するので、VLA の出力は自然と「ハンドオフ円柱に到達して止まる」挙動に収束する。

```python
# record_vla_demos.py
cut = lambda: bool(vp.handoff(env, **hp)[0].item())   # 切り出し点 = §3.4 handoff
...
for _ in range(max_steps):
    action = policy.get_action(env, None)   # vdash_scripted が駆動
    env.step(action)
    if cut():                               # handoff が真 → エピソード確定・export
        reached = True
        break
```

> `--until inserted` にすると「ピック→嵌合まで」の全タスクを記録する別データセット
> （`vla_pick_insert_full`）も作れる。標準は handoff 切りの `vla_pick_handoff`。

### 3.2 ランタイムのハンドオフ（VLA → ルールベースの制御移譲・M6）

`vdash_vla_policy.py` の実行時。本ドキュメントの主題。次節で詳説する。

---

## 4. VLA → ルールベースへの切り替え機構（`VDashVLAPolicy.get_action`）

中心は `isaaclab_arena/policy/vdash_vla_policy.py:190` の `get_action`。

```python
def get_action(self, env, observation):
    self._ensure(env)                                  # 初回のみ後段・IK・述語パラメータを構築
    import isaaclab_arena_environments.mdp.vdash_predicates as vp
    uenv = env.unwrapped

    # --- 前段: VLA（カメラ＋状態から 8-dof 関節目標）---
    pick_act = self._vla.get_action(env, observation)  # (N, 8)
    # --- スイッチ: handoff 述語で env ごとにラッチ ---
    self._in_insertion = self._in_insertion | vp.handoff(uenv, **self._hp)

    # --- 後段: ルールベース嵌合（IK→関節ブリッジ）---
    ins_ik  = self._ins.step(uenv, active=self._in_insertion)  # (N, 7) 相対IK
    ins_act = self._ik_to_joint(uenv, ins_ik)                  # (N, 8) 関節目標

    # --- env ごとに前段 / 後段の出力を選択 ---
    return torch.where(self._in_insertion.unsqueeze(-1), ins_act, pick_act)
```

切り替えを支える4つのポイント：

### 4.1 ラッチ（単調・不可逆・env 独立）

```python
self._in_insertion = self._in_insertion | vp.handoff(uenv, **self._hp)
```

- `_in_insertion` は `(num_envs,)` の bool テンソル。reset 時に全 `False`。
- 毎ステップ述語を OR で積算するので、一度 `True` になった env は **その後ずっと後段に留まる**
  （`reset` まで戻らない）。嵌合中に把持が一瞬緩んで述語が偽になっても、後段に固定されたまま。
- env ごとに独立・ベクトル化。env A は前段・env B は後段、が同時に成立する。

### 4.2 なぜ「述語」で切り替えるのか（`vdash_scripted` との対比）

| | トリガ | 取得元 |
|---|---|---|
| `vdash_scripted` | `pick.step()` が返す `finished` フラグ | **ポリシー内部の状態機械**（TRANSPORT→DONE 到達） |
| `vdash_vla` | `vp.handoff(env, ...)` | **シミュレータの物理状態** |

スクリプト専門家は内部に明示的な状態機械（`APPROACH→…→DONE`）を持つので「終わったか」を
自分で答えられる。一方 VLA はブラックボックスで内部に終了フラグがない。そこで **「終わったか」を
ポリシーに尋ねず、環境の観測量（把持・口元上の静止）から判定** する。データ記録の切り出し点
（§3.1）と同じ述語を使うため、学習分布とランタイムの切り替え点が原理的に一致する。

### 4.3 後段の `active` ゲート

```python
ins_ik = self._ins.step(uenv, active=self._in_insertion)
```

`InsertionController.step(env, active)` は **`active` が真の env でのみ内部状態を更新** する
（フェーズ進行・スパイラル半径・プレス目標などすべて `active` マスクでゲート）。
よって後段は全 env で毎ステップ呼ばれるが、ラッチ前の env では何も進まない。
前段（VLA）と後段（嵌合）の計算は毎ステップ両方走り、出力だけ `torch.where` で選ぶ構造。

### 4.4 アクション空間ブリッジ（IK → 関節）

ここが `vdash_scripted` には無い、VLA 評価固有の実装。

- 評価環境は `franka_joint` 体 → アクションは 8-dof 関節位置 `[7 arm + 1 finger]`
  （`run_eval_grid.py` は VLA のとき `--embodiment franka_joint` を付ける）。
- VLA はこの空間で学習済み（§5 参照） → `pick_act` はそのまま使える。
- しかし `InsertionController` は 7-vector の **相対 IK** `[dx,dy,dz,ax,ay,az,grip]` を出す。
  → `_ik_to_joint`（`vdash_vla_policy.py:171`）で 8-dof 関節へ変換する。

`_ik_to_joint` の中身（記録環境の `FrankaIKJointRecordingAction` と同一の DLS IK を再現）：

1. 姿勢デルタを `_IK_SCALE = 0.5` でスケール（`FrankaActionsCfg` の arm スケールに一致）。
2. EE フレーム姿勢・ヤコビアン（ツールオフセット `panda_hand + (0,0,0.107)` を補正）を計算。
3. `DifferentialIKController`（`command_type="pose"`, `use_relative_mode=True`, `ik_method="dls"`）
   で腕7関節の目標を解く。
4. グリッパは2値 IK（`grip >= 0` → 開 0.04 / 閉 0.0）を指関節目標へマップ。
5. `[arm_des(7), grip(1)]` を結合し `(N,8)` を返す。`torch.where` のため前段・後段とも `(N,8)`。

> **切り分け用のデバッグ経路**: 環境変数 `VDASH_VLA_SCRIPTED_PICK` を立てると前段を VLA の代わりに
> スクリプト専門家ピックに差し替えられる（IK→関節ブリッジ単体の検証用 — 把持が確実なのに嵌合が
> 失敗するならブリッジ側のバグ）。`VDASH_VLA_FORCE_INSERT` は step 0 から全 env を後段に固定する。

---

## 5. VLA 前段（`Gr00tClosedloopPolicy`）とアクション空間の変遷

`isaaclab_arena_gr00t/policy/gr00t_closedloop_policy.py`、設定は
`isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml`。

| 項目 | 値 |
|---|---|
| モデル | GR00T N1.6（`datasets/vdash/vla_pick_handoff` で微調整） |
| 出力 | 8-dof 関節位置目標（`processed_actions` 空間） |
| `action_horizon` / `action_chunk_length` | 16 / 16（チャンク推論、`ActionChunkingState`） |
| カメラ | `wrist_cam_rgb` / `left_cam_rgb` / `right_cam_rgb`（256×256×3） |
| 状態 | `joint_pos`(9) / `eef_pos`(3) / `eef_quat`(4) / `gripper_pos` |
| 言語指示 | "Pick up the peg and move it over the socket."（口元まで運ぶところまで） |
| 体（embodiment） | `franka_joint`（8-dof 関節位置アクション） |

### アクション空間の変遷（M9 凍結 → 現行 v2/v3）— 重要

`docs/vla_interface.md`（M9 で凍結）が定義したアクション空間は **7-D 相対 IK**
（`[dx,dy,dz,ax,ay,az,gripper]`, scale=0.5, 60 Hz）で、デモも 7-D で記録されていた。
その後 VLA 学習データは **`processed_actions` = 8-dof 関節位置** に切り替わった
（`vdash_pick_handoff_v2/v3_config.yaml`）:

- `action_name_sim: "processed_actions"`、`action_joints_config_path: 8dof_action_space.yaml`
  （= `panda_joint1..7` + `panda_finger_joint1`、計 8）。
- 記録環境の `FrankaIKJointRecordingAction` が、専門家の 7-D IK コマンドを DLS IK で解いて
  関節目標に変換し、`processed_actions` として保存する。VLA はこの **関節空間** を直接学習。
- v3 は **角型グリップのペグ＋ヤウ整合把持**（`--arm_init_std 0.15` で開始姿勢を広く乱択）で
  200 デモ。v2/v3 は円柱ペグの v1 とアセット非互換。

この変遷の結果、現行の `vdash_vla` 評価は `franka_joint`（8-dof 関節）で動き、`vdash_vla_policy.py`
の「VLA = 8-dof 関節を直接出力、後段の 7-D IK は DLS IK で 8-dof へ橋渡し」という設計に一致する。
**観測キー・ハンドオフ境界・言語テンプレートは凍結契約のまま**（変わったのはアクション表現のみ）。

---

## 6. 後段ルールベース嵌合（`InsertionController`）

`isaaclab_arena/controllers/insertion_controller.py`。フェーズ機械
`SETTLE → PRESS → RELEASE → DONE`（詳細・実測は `docs/milestones/M3.md`）。

| フェーズ | 動作 |
|---|---|
| **SETTLE** | ペグ先端を socket 軸上・口元少し上に持っていき、EE 速度がしきい以下になるまで静止待ち。接近運動量を殺し、§3.4 `handoff` を確実に発火させる |
| **PRESS** | 低力サーボ（微小 Fz 目標を追従）で先端を降下。捕捉前は xy にアルキメデス **スパイラルサーチ**、任意で **RCC**（並進・回転コンプライアンス）。詰まり検知で SETTLE へリトライ |
| **RELEASE** | 先端が口元より深く着座したらグリッパを開いて退避 |

### 6.1 §2.1 入力制約（重要）

後段が使ってよい入力は **socket 姿勢（既知のジグ固定治具）＋ EE 自己受容感覚
（`tip_estimate`：把持公称の先端推定）＋ 手首 F/T センサモデル** のみ。
**ペグの真値姿勢は決して読まない**（読んでよいのは述語と §3.5 ロガーだけ）。
制御は **先端相対**: ペグは剛体把持されているので EE を Δ 動かせば先端も Δ 動く
→ EE 目標 = `現EE + (先端目標 − 先端推定)`。

> M3 の知見: 真値姿勢フィードバックを外した §2.1 準拠化 **そのもの** が成功率を 68–74% → 100%
> に上げた（傾き追従のリトライ暴走が消えた）。接触力は低力サーボで ~35N → 中央値 4.25N に低減。

### 6.2 主な閾値（`configs/vdash/controllers.yaml` の `insertion:`）

| パラメータ | 値 | 意味 |
|---|---|---|
| `use_settle` | true | SETTLE フェーズ ON |
| `settle_height` | 0.035 m | SETTLE の先端ドウェル高さ（§3.4 円柱内に収める） |
| `use_force_servo` | true | 低力サーボ ON |
| `f_target` | 4.0 N | サーチ中のプレス荷重目標 |
| `seated_depth` | 0.018 m | 着座とみなす深さ → RELEASE |
| `spiral_radius_max` | 0.008 m | スパイラル半径上限 |
| `jam_force` / `retry_after_steps` / `max_retries` | 12 N / 120 / 6 | 詰まり検知・リトライ |
| `rcc_xy_gain` / `rcc_rot_gain` | 0.0 / 0.0 | RCC（c=2.0 では中立につき OFF。実装は利用可能・Whitney 1982） |

> クリアランスの限界（M5）: c=2.0 では L0–L3 で 98–100% だが、c≤1.0 で 0/24 に崖落ち
> （funnel→bore リムで楔詰まり）。位置制御の剛体 EE では機械的楔を外せず、能動コンプライアンスか
> 「hop-and-search」リカバリが必要、というのが既知の課題。

---

## 7. 実行方法（M6 VLA 評価）

```bash
# 単一 env（非タイルのカメラは 1 env をレンダリング）
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena.policy.vdash_vla_policy.VDashVLAPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml \
  --num_envs 1 --enable_cameras --headless --num_episodes 20 \
  vdash_pick_insert --embodiment franka_joint --clearance 2.0 --level L1"

# グリッド評価（vdash_scripted / vdash_rule_based / vdash_vla を横断）
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py --policies vdash_vla --clearances 2.0 --levels L1
```

---

## 8. 設計上の要点（まとめ）

1. **切り替えは「ポリシーに訊く」のではなく「環境状態で決める」。**
   VLA は終了フラグを持たないので、観測可能な物理量（把持・口元上の静止）の述語で判定する。
2. **学習の切り出し点とランタイムの切り替え点が同一述語。**
   `record_vla_demos.py --until handoff` と `VDashVLAPolicy` のラッチは同じ `vp.handoff`
   （凍結契約 `vla_interface.md` が明記）。→ VLA はハンドオフ円柱に到達して止まる挙動を自然に学ぶ。
3. **ラッチは単調・不可逆・env 独立。** 一度後段に入った env は reset まで戻らない。
4. **後段は §2.1 準拠で再利用。** 3ポリシー（scripted / rule_based / vla）が同一の
   `InsertionController` を共有し、前段と切り替えトリガだけが異なる。
5. **アクション空間ブリッジは VLA 評価固有。** 後段の相対 IK 出力を、記録環境と同一の DLS IK で
   `franka_joint` の 8-dof 関節目標へ変換して整合させている（M9 凍結の 7-D IK から、現行は
   `processed_actions` = 8-dof 関節へ移行済み）。

---

### 参照ファイル

| 役割 | パス |
|---|---|
| VLA→ルール スイッチ本体 | `isaaclab_arena/policy/vdash_vla_policy.py` |
| §3.4 述語（`handoff` / `grasped` / `inserted`） | `isaaclab_arena_environments/mdp/vdash_predicates.py` |
| 後段 嵌合コントローラ | `isaaclab_arena/controllers/insertion_controller.py` |
| スクリプト専門家ピック | `isaaclab_arena/controllers/scripted_pick.py` |
| 前段 VLA（GR00T closed-loop） | `isaaclab_arena_gr00t/policy/gr00t_closedloop_policy.py` |
| VLA 評価設定 | `isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml` |
| 述語/幾何 閾値 | `configs/vdash/task.yaml` |
| コントローラ ゲイン | `configs/vdash/controllers.yaml` |
| 教師データ記録（切り出し点） | `scripts/vdash/record_vla_demos.py` |
| データセット設定（v2/v3 = `processed_actions` 8-dof） | `isaaclab_arena_gr00t/lerobot/config/vdash_pick_handoff_v{2,3}_config.yaml` |
| 比較ポリシー（scripted / rule_based） | `isaaclab_arena/policy/vdash_scripted_policy.py` / `vdash_rule_based_policy.py` |
| グリッド評価ランナー | `scripts/vdash/run_eval_grid.py` |
