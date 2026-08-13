# B-DASH 開発レポート（M1〜M3）

対象: RSJ 2026（9月）論文の実験基盤。VLA（知覚・搬送）+ ルールベース（精密嵌合）の階層制御を
Isaac Sim 5.1 / Isaac Lab Arena 上に構築する。本レポートは **M1〜M3** の作業内容を「何を・なぜ・
どうやって」で記録する（M0 = Forge 物理パラメータ抽出は別途 `docs/milestones/M0.md`）。

実行環境: ローカル RTX 5080 / Docker `isaaclab_arena-latest` / `/isaac-sim/python.sh`。
ヘッドレス実行時は `unset DISPLAY`（起動 7分→12秒）。

---

## M1 — パラメトリック・アセット生成 + 物理安定性検証 ✅

### 何を
クリアランス `c ∈ {2.0, 1.0, 0.5, 0.25} mm` の **socket（穴）** と **peg（ペグ）** を手続き生成し、
Forge 級の接触物理で「保持ペグの降下挿入」が貫通・爆発なく成立することを確認した。

![peg/socket 断面](images/bdash/cross_section.png)
*peg / socket 断面（左）と口元ファネルの拡大（右）。peg は Ø8mm シャフト + Ø20mm 円柱グリップ、
socket は深さ25mm のブラインド穴 + 3mm リードインファネル。クリアランス c はボア半径 = 4+c で効く。*

![クリアランス系列](images/bdash/clearance_series.png)
*クリアランス系列（縮尺一致）。青=peg、灰=ボア。c=2.0→0.25mm でラジアル隙間が狭くなる。*

- `scripts/bdash/generate_assets.py`: trimesh の CSG（manifold エンジン）で peg / socket メッシュを
  生成 → STL → `MeshConverter` で USD 化。
  - peg: 先端チャンファ（C0.5）+ Ø8mm シャフト（長50mm）+ グリップ。**先端を local z=0** に置く。
  - socket: 60×60×30mm ブロックに深さ25mm のブラインド穴（半径 = `peg_r + c`）+ 口元のチャンファ。
  - **穴側に SDF メッシュ collision（解像度1024）**、ペグ側は convexDecomposition。
- `configs/bdash/peg_insert/assets.yaml`: 全寸法・クリアランス系列・collision パラメータを外出し。
- `isaaclab_arena_environments/mdp/env_callbacks.py`: Forge 物理コールバック
  `bdash_assembly_env_cfg_callback`（dt=1/120, TGS solver, max_position_iteration_count=192,
  contact_offset=0.0002, friction=1.0, gpu_max_num_partitions=1）。

### なぜ
- クリアランス掃引が論文の主要図の片翼であり、`c` をパラメトリックに変えられる穴が必須。
- 既存リポジトリの `tabletop_peginsert_environment` は **(1) 単一固定の peg/hole**、
  **(2) fixed/held が逆**（固定すべき穴が held になっていた）、**(3) §3.4 述語なし** で、そのままでは
  研究要件を満たさない（→ 拡張ではなく新規生成系で対応）。
- 穴の内面は鋭いエッジを持つため、convex 近似では嵌合が破綻する。**SDF mesh collision が必須**。

### どうやって / 主要な技術判断
- `MeshConverter` は入力 STL のハッシュでキャッシュするため、再生成時は force 変換 + コンテナ内で
  root 所有の旧アセット削除が必要、という運用を確立。
- 微小・軽量（19g）ペグの素のインピーダンス制御は数値発散（ω·dt≈2.3）するため、M1 の安定性検証は
  **重い実効質量 + 手首クランプ + 重力駆動降下** の簡易ハーネスで実施。

### 結果 / 残課題
- 4 クリアランス全てで **挿入が幾何学的に成立**（c=0.25 でも降下 ~30mm）。
- 軸ズレ初期条件での **SDF 由来の力スパイク（爆発）** が SDF 解像度（256/512/1024）に依らず残存。
  ユーザ判断で「**M3 の力制限コントローラ下で検証する既知リスク**」として受理（`docs/milestones/M1.md`）。

---

## M2 — Arena タスク統合（述語→終了条件 + §3.5 ログ）✅

### 何を
§3.4 の局面述語を **実際の Arena タスクの終了条件として配線**し、ペグに 2 つの ContactSensor を付け、
全閾値を YAML 外出しし、**§3.5 のエピソード単位ログ**（述語成立時刻・ハンドオフ幾何・力統計・終端原因）を
出力し、環境を CLI 登録した。

| ファイル | 役割 |
|---|---|
| `mdp/bdash_peg_predicates.py` | §3.4 述語: `grasped` / `handoff` / `inserted` / `force_violation` / `insertion_failed`。torch のみ・sim 非依存。 |
| `tasks/bdash_pick_insert_task.py` | socket=固定(kinematic)・peg=held。終了条件 `success=inserted` / `insertion_failed=力超過` / `object_dropped` / `time_out`。 |
| `metrics/bdash_handoff_logger.py` | `RecorderTerm` でステップ毎に状態蓄積→reset 前に 1 エピソード 1 行の JSONL を書く。 |
| `mdp/bdash_peg_randomization.py` | シーン多様性 L0–L3 プロファイル（姿勢／照明／テクスチャ／ディストラクタ）。 |
| `bdash_pick_insert_environment.py` + `cli.py` | 環境合成と登録（`--clearance`/`--level`/`--preinsert`）。 |

### なぜ
- 既存 `AssemblyTask` の成功判定は **根位置の近接のみ**で、述語も力ログもハンドオフ解析もない。
  本研究は「成功率だけでなく**述語ログ・終端誤差ログの確実な記録**」が一級要件。

### どうやって / 設計上のポイント
- 述語は `func(env, **params)` 形式なのでそのまま `TerminationTermCfg` に差せる。
- **scene キーは登録アセット名**（`bdash_peg`, `bdash_socket_c1`…）でビルド毎に変わるため、
  タスク側で YAML のプレースホルダを実アセット名に上書き（キャッシュは破壊しないよう deepcopy）。
- アクティブな socket の **クリアランス c を `inserted` 判定にビルド時注入**。

### M2 で踏んだバグ（教訓）
1. **scene キー不一致** — 述語が `peg`/`socket` を参照していたが実キーは登録名。タスクで上書きして解消。
2. **L0 が実は固定でなかった** — 空レンジでも at-reset ランダマイザが走り、ペグを原点付近（テーブル貫通、
   2289N）へ再配置。L0 ではランダマイザ項を `None`（EventManager がスキップ）にし、
   `reset_scene_to_default` で正準姿勢を保持する方式に修正。
3. **オプション event 項は宣言フィールド必須** — `combine_configclass_instances` は宣言済み dataclass
   フィールドしかコピーしないため、L2/L3 の照明・テクスチャ項を動的 setattr ではなく宣言（既定 None）に。

### 結果（実機検証・ヘッドレス）
- 述語ユニットテスト合格（`BDASH_PREDICATES_OK`）。
- **成功経路**（c=2.0, `--preinsert`）: ペグが綺麗に着座（深さ21.8mm, 0N）→ `inserted` → **result=success**、
  60/60 エピソード成功。`t_inserted_s` 等を JSONL に記録。
- **失敗経路**（c=1.0）: 狭い SDF 穴へのテレポート由来の力スパイク → `insertion_failed`。
- 全閾値が YAML 由来であることを cfg ダンプで確認。

---

## M3 — スクリプト・エキスパート（把持→搬送）+ ルールベース嵌合 ◐（結合動作は完成、調整中）

### 何を
特権情報（ペグ・穴の真値姿勢）を用いる **2 つのルールベース制御器** を実装し、Franka の relative-IK で駆動。

- `controllers/ee_control.py`: 目標 EE 世界姿勢 → 7次元 relative-IK アクション
  （位置は world 差分、姿勢は world 軸の axis-angle、グリッパは二値）。`level_to_down()` で
  グリッパを厳密に真下へ整える補助関数を含む。
- `controllers/scripted_pick.py`: `approach → descend → grasp → lift → transport` の状態機械。
  VLA フロントエンドの代役（学習はスコープ外）。
- `controllers/insertion_controller.py`: 力制御 + スパイラルサーチ。`ALIGN → PRESS → RELEASE`。
  **tip 相対制御**（剛体把持なので ΔEE = Δtip）。
- `policy/bdash_scripted_policy.py`: pick → insertion を逐次接続（pick 完了で insertion にラッチ＝ハンドオフ）。
  policy `bdash_scripted` として登録。
- `configs/bdash/peg_insert/controllers.yaml`: ゲイン・閾値を全て外出し。

### なぜ
- ルールベース嵌合（力制御 + スパイラル）が階層制御の精密側であり、M5 のクリアランス曲線・
  M6 のアブレーションを取るための土台。受け入れ基準は **L1・c=2.0 で成功率 ≥80%（50試行）**。

### どうやって — 反復デバッグの記録（M3 の本質）
実機反復で次々と根本原因を特定・修正した。各々が独立した failure mode だった。

1. **アームが遅すぎる（~0.04 m/s）** → 1 エピソードが時間内に終わらない。
   原因: assembly 用 Franka cfg が stiffness を 150 に**下げて**いた。
   対策: B-DASH 専用 cfg（`FRANKA_PANDA_BDASH_CFG`, stiffness 400 + 接触センサ）。コンプライアンスは
   アームの柔さではなく **コントローラの力制限**で与える方針に。

2. **ペグの傾き暴走（5°→61°）でジャム** → 把持が約4.8°傾いて保持していた（レディ姿勢の向きが
   真下でない）。対策: 保持姿勢を `level_to_down()` で**厳密な真下**へ。これで L0/c=2.0 は傾き 0.4° で
   **完全成功**（深さ・力・ハンドオフ幾何まで §3.5 に記録）。

3. **L1（姿勢ランダム化）でペグに届かない** → `sample_object_poses` は env 原点基準の**絶対座標**を
   サンプルするため、±10cm がロボット土台付近（到達不可）になっていた。
   対策: ランダム化レンジを**作業空間（~0.5, 0 中心）の絶対ボックス**に修正 + 最小離間 0.12m。

4. **L1 で把持時にペグが傾く** → 平行グリッパが **正方プリズムのグリップを角接触**で掴み、ランダム yaw で
   傾く。対策: **グリップを円柱に変更**（yaw 不変な把持）。peg アセットを再生成。
   → L1/c=2.0 単発で傾き ≤4°、挿入成功を確認。

5. **挿入が口元でジャム（残課題の中心）** → 50試行評価で段階的に改善:
   - 軽い押付け + スパイラル（横滑りで穴口を探索） + リトライ: 62% → 74%（force violation が 7→1 に激減）。
   - 二段押付け（捕捉前は軽く、捕捉後は速く本締め） + §3.5 に最大深さ・最終傾きを追加記録。
   - **傾きトリガのリトライ**（傾く前に引き上げ再鉛直化）: 失敗時の傾きは 5–7° に収束したが、
     再押付けで同じ ~7° に再発しループ → 68%。
   - **口元に円錐ファネル（座ぐり 3mm）追加**: 平面点接触ではなく斜面で tip を芯出しする狙い。
     → **効果なし（68%）**。失敗時の傾き 4–7°・接触力 30–39N は不変で、**成功例も同じ ~35N** だった。
     つまり「ファネルの有無」ではなく **接触力スパイクそのもの** が支配要因だと判明。

### 現状 / 残課題（最新の診断）
- **把持・搬送・ハンドオフは 100% 安定**（50/50 で把持成功）。L0/c=2.0 は安定成功。
- **L1/c=2.0 は約 68–74%**（目標 80% に未達、調整中）。
- **真因（判明）**: 硬い位置制御アームが **リム接触時に力制限の反応より速く ~35N を発生**させ、その
  オフセット接触のモーメントでペグが数度傾く → ジャム。成功と失敗の差は力の大小ではなく
  「接触が芯に当たるか／オフセットして傾くか」という確率的な分岐。
- **次の一手（方針確定）**: ファネルや事後リトライではなく **接触を優しくする**こと。
  (1) ALIGN 末で速度を落とし口元上で一旦静定（インパクト除去）、(2) PRESS を **低力サーボ化**
  （目標 ~3–5N で毎ステップ即応に上げ下げ）して接触力自体を一桁下げる。これにより傾きモーメントを
  根本から小さくする。
- 補足: 現状ハンドオフ述語（`speed<0.05` を要求）は高速降下中に成立しないため `t_handoff` が null。
  ハンドオフ幾何のログ取得には降下を一旦ホバーさせる小改修が必要（M3 完了時に対応予定）。

### 成功率の推移（L1, c=2.0, 50試行, seed=0, 8並列）
| 施策 | 成功率 |
|---|---|
| 円柱グリップ + 基本の力制御 | 62% |
| + 軽押付け・スパイラル・リトライ | 74% |
| + 二段押付け | 70% |
| + 傾きトリガ・リトライ | 68% |
| + 口元ファネル | 68%（効果なし→真因は接触力スパイクと判明） |
| 低力サーボ化 + 静定（次の施策） | 実装予定 |

![M3 成功率の推移](images/bdash/m3_success_progression.png)
*L1 / c=2.0mm / 50試行での成功率推移。74% がこれまでの最高。目標 80%（破線）に未達。*

![peg 3D](images/bdash/peg_3d.png)
*生成された peg（円柱グリップで yaw 不変な把持を実現）。*

---

## 実行方法（再現コマンド）

すべてコンテナ `isaaclab_arena-latest` 内で実行する。インタプリタは `/isaac-sim/python.sh`。
ヘッドレス実行では起動高速化のため **`unset DISPLAY`**（7分→12秒）。
作業ディレクトリは `/workspaces/isaaclab_arena`。

### ① GUI で動かして自分でスクショする（WebRTC ライブビュー）
`LIVESTREAM=2` で起動すると WebRTC でブラウザから視聴・スクショできる（`--headless` のまま映像だけ配信）。
**この場合は `unset DISPLAY` しない**。`--num_envs 1` 推奨。

```bash
# スクリプト・エキスパート（pick → 嵌合）を 1 環境で実行し、ブラウザでライブ視聴
docker exec -it isaaclab_arena-latest bash -lc '
  cd /workspaces/isaaclab_arena && \
  LIVESTREAM=2 /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type bdash_scripted --num_steps 900 --headless --livestream 2 --num_envs 1 \
    bdash_pick_insert --clearance 2.0 --level L0'
# 起動後、表示される WebRTC URL（既定 http://<host>:8211 等）にブラウザで接続してスクショ。
```

- 把持の様子を見たいなら `--level L0`（固定配置）、ランダム化を見たいなら `--level L1`〜`L3`。
- ペグが穴に刺さった「成功状態」を即見たいなら **`--preinsert`**（ペグを最初から穴に着座させる）:
  ```bash
  LIVESTREAM=2 /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 200 --headless --livestream 2 --num_envs 1 \
    bdash_pick_insert --clearance 2.0 --level L0 --preinsert
  ```

### ② M1: アセット生成
```bash
docker exec isaaclab_arena-latest bash -lc 'cd /workspaces/isaaclab_arena && unset DISPLAY && \
  /isaac-sim/python.sh scripts/bdash/generate_assets.py --headless'        # peg + socket 全て
# 一部だけ再生成（高速）: peg のみ / socket のみ
#   /isaac-sim/python.sh scripts/bdash/generate_assets.py --peg-only   --headless
#   /isaac-sim/python.sh scripts/bdash/generate_assets.py --socket-only --headless
```
出力: `assets/bdash/peg_insert/{peg,socket_c*}.usd`（root 所有。再生成時はコンテナ内で旧ファイル削除）。

### ③ M2: 述語・ログの検証
```bash
# 述語ユニットテスト（Isaac 不要・数秒）
docker exec isaaclab_arena-latest bash -lc 'cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena_environments/tests/test_bdash_predicates.py'   # -> BDASH_PREDICATES_OK

# 成功経路 + §3.5 ログ出力（preinsert で inserted→success を確認）
docker exec isaaclab_arena-latest bash -lc 'cd /workspaces/isaaclab_arena && unset DISPLAY && \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 60 --headless \
    bdash_pick_insert --clearance 2.0 --level L0 --preinsert --run_tag demo'
# ログ: logs/bdash/bdash_handoff_*.jsonl（1 エピソード 1 行）
```

### ④ M3: スクリプト方策のヘッドレス評価（成功率）
```bash
docker exec isaaclab_arena-latest bash -lc 'cd /workspaces/isaaclab_arena && unset DISPLAY && \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type bdash_scripted --num_episodes 50 --headless --num_envs 8 --seed 0 \
    bdash_pick_insert --clearance 2.0 --level L1 --run_tag eval'
# 成功率は logs/bdash/bdash_handoff_eval_*.jsonl の result 集計で確認。
# ステップ毎の局面・力・傾きを見たい時は環境変数 BDASH_DEBUG=1 を付与。
```

主な CLI 引数: `--clearance {2.0|1.0|0.5|0.25}`（mm）, `--level {L0|L1|L2|L3}`,
`--preinsert`（着座状態で開始）, `--num_envs N`（並列）, `--num_episodes / --num_steps`,
`--seed`, `--run_tag <name>`（ログ名）。

### ⑤ レポート図の生成
```bash
docker exec isaaclab_arena-latest bash -lc 'cd /workspaces/isaaclab_arena && unset DISPLAY && \
  /isaac-sim/python.sh scripts/bdash/make_report_figures.py'   # -> docs/images/bdash/*.png
```

---

## 横断的に効いた知見
- **特権スクリプトでも「把持の鉛直性」と「口元のリードイン」が支配的**。傾き 1° 以下なら緩い
  クリアランスはほぼ確実に入るが、リム接触の offset で数度傾くと即ジャムする。
- **位置制御の硬いアームでの "力制御" は本質的にトレードオフ**: 速度（探索効率）と接触の優しさ（傾き回避）。
  目標リード距離のクランプ + 力バンド + 二段化 + リトライで近づけている。
- 研究的整合性: ファネルは**穴の入口のみ**を広げ、**クリアランス c（掃引変数）と深部の嵌合難度は不変**。
  ルールベースが精密嵌合を得意とする筋書きと矛盾しない。

## 成果物（ファイル）
- アセット: `scripts/bdash/generate_assets.py`, `configs/bdash/peg_insert/assets.yaml`, `assets/bdash/peg_insert/*.usd`
- タスク/述語/ログ: `isaaclab_arena/tasks/bdash_pick_insert_task.py`,
  `isaaclab_arena_environments/mdp/bdash_peg_predicates.py`, `isaaclab_arena/metrics/bdash_handoff_logger.py`
- コントローラ: `isaaclab_arena/controllers/{ee_control,scripted_pick,insertion_controller}.py`,
  `isaaclab_arena/policy/bdash_scripted_policy.py`, `configs/bdash/peg_insert/controllers.yaml`
- 環境/設定: `isaaclab_arena_environments/bdash_pick_insert_environment.py`,
  `configs/bdash/peg_insert/task.yaml`, `mdp/bdash_peg_config.py`, `mdp/bdash_peg_randomization.py`
- 記録: `docs/milestones/M0.md` … `M2.md`, 本レポート
