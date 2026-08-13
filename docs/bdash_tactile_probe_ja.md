# B-DASH 触覚プローブ（bore-edge tactile probe）— 設計ノート

**状態:** 実装完了・検証待ち（2026-06-27）。GPU 解放後に scripted テストベッドで検証予定。
コードは `isaaclab_arena/controllers/insertion_controller.py`（コメントは英語＝コードベース慣習）。
全体計画は `docs/bdash_precision_budget_plan.md` を参照。

## 1. 背景と目的

M11 で、B-DASH の挿入成功率を 20%（4/20）に縛っているのは VLA の **グリッパ内把持誤差**（ペグが
グリッパ内で約 8°/9mm 傾いて掴まれている）であり、§2.1 の「ペグ姿勢ブラインド」挿入コントローラは
これを構造的に補正できない、と特定した（チート診断：真の先端を与えると 8mm オフセットで 0→55%）。

本プローブの目的は、**ペグの真の姿勢を一切読まずに**（§2.1 を守ったまま）グリッパ内の先端誤差
`e = true_tip − tip_estimate` を**能動的な接触探索で推定**し、`tip_estimate` を補正して挿入を回復する
こと。VLA の再学習は不要。これは「理論が処方する interface-widening（インタフェースの拡張＝
ブラインドチャネルを能動センシングで開く）」の具体実装であり、論文の主張（精度バジェット理論）の
**実証**にあたる（推定器そのものは TacGraph 等の先行研究の系譜で、新規性は理論側）。

## 2. §2.1 で使ってよい信号だけを使う

コントローラが使えるのは **固有受容（`ee_frame` → `tip_estimate`）／既知のソケット治具姿勢／手首
力覚モデル `_wrench`（ペグ接触センサの合力）** のみ。ペグの真姿勢 `peg.root_*` は読めない。
プローブはこの 3 つだけで動く（`_calibrate_step` 内にペグ姿勢の参照は一切なし。チート診断
`BDASH_CHEAT_TIP` のみがゲート内で真姿勢を使う＝§2.1 違反の上限値専用）。

## 3. 原理 — bore-edge（穴縁）探索

ソケットの穴（bore）は既知のソケット xy を中心とする半径 `bore_r = 0.004 + c` の円（c=2mm で 6mm、
口元に 3mm の 45° 面取り funnel）。コントローラが「想定先端（assumed tip）」をある xy に下ろすと、
**真の先端は `assumed + e_xy`** に行く。つまり「想定先端を下ろしたとき真先端が穴に落ちる」想定 xy の
集合は、`socket_xy − e_xy` を中心とする半径 `bore_r` の円になる。

→ 想定先端 xy のグリッド上で 1 点ずつ「下ろして接触する高さ（想定先端 z）」を記録すると、**真先端が
穴の上に来た点ほど深く沈む（接触 z が低い）**。沈み込み（平坦面プラトーからの落差）で重み付けした
グリッドオフセットの重心が、想定座標での穴中心 ≈ `−e_xy` を与える：

```
e_xy = −Σ_i w_i · grid_offset_i / Σ_i w_i      （w_i = max(penetration_i − margin, 0)）
e_z  = mouth_z − plateau                         （plateau = 平坦面での接触 z = mouth_z − e_z）
```

`self.tip_correction = (e_xy, e_z)` を `tip_estimate` に加算（`insertion_controller.py` の 1 行
`tip = tip_estimate(...) + self.tip_correction`）。以降コントローラは**真の先端**を基準に
深さ・捕捉・スパイラルを回すので、チートと同じ効果を**合法に**得る。

**critique を反映した重要点:** funnel は 3mm しかないのに誤差は 9–17mm なので、cone（円錐）の正弦
フィットは破綻する。本実装は**穴縁の沈み込みをグリッドで直接見る**方式（funnel 幅に依存しない）を
主軸にした。クリアランスが狭い（`bore_r → peg_r`）と沈み込み信号が消えるので、**適用は c≥2.0** を想定。

## 4. 観測可能性（できること・できないこと）

- **lateral（横ずれ `e_xy`）:** 観測可能。穴が横ずれを「接触 z の差」に変換してくれる。
- **tilt（傾き）:** 力のみ（`peg_contact` は `net_forces_w` のみでトルクなし）では先端 1 点から
  傾きは原理的に不可分。→ **位置補正のみ → 約 55%（チート上限）が現実的な天井**。
  傾きを部分的に観測するストレッチ案として、`peg_finger_contact.force_matrix_w`（指ごとの力）の
  グリップ中心まわりのモーメントが §2.1 合法な傾き手がかり（今後）。

## 5. 実装と環境変数

- フェーズに `CALIBRATE` を前置（`CALIBRATE, SETTLE, PRESS, RELEASE, DONE = range(5)`）。
  比較は全て名前ベースなので、**プローブ OFF 時は開始フェーズ＝SETTLE のまま挙動不変**。
- `_calibrate_step()`：ベクトル化された GOTO（安全高へ上昇＋xy 中心合わせ）→ DESCEND（接触＝
  ベースライン超え or フロア到達まで下降、想定先端 z を記録）→ 全 K 点で SOLVE → `tip_correction`
  設定 → `CALIBRATE→SETTLE`。傾いた把持で指の合力ベースラインが非ゼロになるので**毎タッチで
  ベースライン差分**を取る（contact-SNR 対策）。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `BDASH_TACTILE_CAL` | off | プローブ有効化 |
| `BDASH_CAL_HALF_MM` | 12 | グリッド半幅（mm） |
| `BDASH_CAL_STEP_MM` | 6 | グリッド間隔（mm）→ 既定 5×5=25 タッチ |
| `BDASH_CAL_DEPTH_MM` | 8 | 探索フロア（口元からの最大下降, mm） |
| `BDASH_CAL_FTOUCH_N` | 2.0 | 接触判定（ベースライン超過, N） |
| `BDASH_CAL_PENMARGIN_MM` | 2.0 | funnel リップノイズの足切り（mm） |
| `BDASH_CAL_DIAG` | off | env0 の推定 `e` をログ |

関連診断：`BDASH_GRASP_DIAG`（真の `e` を eval 専用で記録＝採点用）、`BDASH_GRASP_OFF_MM`
（テストベッドで把持を意図的にずらす）、`BDASH_CHEAT_TIP`/`BDASH_TIP_NOISE_MM`（上限と精度仕様）。

## 6. 検証手順（GPU 解放後）

1. **精度:** `BDASH_GRASP_OFF_MM` を 0〜12mm で振り、`BDASH_TACTILE_CAL=1 BDASH_CAL_DIAG=1` で
   推定 `e_xy` を `BDASH_GRASP_DIAG`（真値）と比較 → **`e_xy` RMSE** を報告（メイク・オア・ブレイク）。
2. **精度仕様:** `R_widened(σ)` 掃引（`BDASH_TIP_NOISE_MM`）から「何 mm の RMSE で何 % か」を読み、
   プローブのグリッド解像度（タッチ数）を調整。
3. **閉ループ:** scripted テストベッドで `BDASH_TACTILE_CAL=1`、offset 8mm → 0/20 がどこまで回復するか
   （目標：チート 55% に肉薄）。
4. **回帰:** `BDASH_TACTILE_CAL` 無効で scripted が 98% のまま（既定パス不変）を確認。
5. **VLA:** `bdash_vla_v6_recovery` に適用し、20% → どこまで上がるか（再学習なし）。
6. **§2.1 監査:** `grep root_pos_w/root_quat_w` で制御経路にペグ姿勢が無いこと（チートゲート内のみ）。

## 7. 期待値と限界

offset 8mm のとき：ブラインド 0/20 → 位置補正で **〜45–55%**（推定 RMSE 次第）。VLA（〜9mm/8°）では
20% → **〜40–50%**（再学習なし、約 2–2.5 倍）。**物理的天井は約 55%**（ペグが物理的に軸ずれのまま
なので、先端を完全に知っても限界）。これを超えるには傾き観測（finger force-matrix）か再把持
（バイナリグリッパでは不可）か VLA の把持品質向上が必要 — これは論文の「理論が予測する天井」として提示。
