# server B: S3 チャック微調整（8/26朝の一発評価用）— コピペ用

```bash
# 1) データ取得（HFプライベート、umeganでログイン済み前提）
hf download umegan/bdash-chuck-full247 --repo-type dataset \
   --local-dir datasets/bdash/chuck_full247/lerobot

# 2) 一発微調整（過去v4/v6と同じ検証済みレシピ、既定=projector+diffusionのみ）
docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc '
  cd /workspaces/isaaclab_arena &&
  DATASET_PATH=datasets/bdash/chuck_full247/lerobot \
  OUTPUT_DIR=models/bdash-gr00t-n1-7-chuck-full250 \
  nohup bash scripts/bdash/train_vla.sh > logs/bdash/train_chuck_full250.log 2>&1 &'

# 3) 完了後、重みをHFへ（ローカルで8/26朝評価に引く）
hf upload umegan/bdash-gr00t-n1-7-chuck-full250 \
   models/bdash-gr00t-n1-7-chuck-full250 . --private
```

- データ: 247デモ(監査で偽成功3本除外済み)・直立のみ・高速教師・エピソード毎色言語(red/gold/silver/blue、
  held-out色は不使用)・実測30fps・3視点(ego/left/right)
- 評価はローカルで1回だけ（キルスイッチ）: 合格→デモ通し、不合格→教師のみで発表確定

---
## 追加（任意だが推奨）: lift スライスの並行学習 — 階層デモ用

full247 と同一軌道を「持ち上げ完了」で切った 247 本（`umegan/bdash-chuck-lift247`、
言語は "…and lift it clear."）。**運用の分業（VLA=ピックのみ）に一致**し、ペグ時代の教訓
「階層には handoff 学習モデル」に対応する。GPU が2枚あれば並行、1枚なら full → lift の順で夜通し。

```bash
hf download umegan/bdash-chuck-lift247 --repo-type dataset \
   --local-dir datasets/bdash/chuck_lift247/lerobot
docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc '
  cd /workspaces/isaaclab_arena &&
  DATASET_PATH=datasets/bdash/chuck_lift247/lerobot \
  OUTPUT_DIR=models/bdash-gr00t-n1-7-chuck-lift247 \
  nohup bash scripts/bdash/train_vla.sh > logs/bdash/train_chuck_lift247.log 2>&1 &'
hf upload umegan/bdash-gr00t-n1-7-chuck-lift247 \
   models/bdash-gr00t-n1-7-chuck-lift247 . --private
```

8/26朝の使い分け: **E2E通しデモ → full モデル / 階層デモ(VLAピック→ゲート→古典挿入) → lift モデル**。
