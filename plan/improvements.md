# face-blurrer 改善計画

## 背景・制約

- 行灯（青森ねぶた的なもの）が写った祭り写真から、実際の人の顔だけをぼかすツール。
  行灯の人形の顔はぼかさない。
- 顔が大きい・画像上半分にあるなどの「行灯疑い」は人間が y/n で判断する（自動判定の
  誤りコストが高いため、人の確認を挟む方針は維持する）。
- 一度に 1000 枚程度を M2 Mac (ノート PC) で処理する。重すぎる処理は不可。
- 写真の EXIF はいじりたくない（→ 現状は cv2.imwrite で EXIF が全消失しているので修正が必要）。
- **写真の品質はできるだけ下げずに保存したい。解像度は 4000〜6000px 程度を想定。**

## 改善点一覧（実装順）

### 1. パイプラインを 3 フェーズに分離（抜本的変更・最重要）

現状は検出（機械・重い）・判断（人間・対話）・加工（機械・軽い）が 1 つのループに
絡み合い、途中結果が残らない。これを中間成果物を挟んだ 3 段階に分離する：

```
detect:  in/ ──(モデル・無人・重い)──→ work/detections.json（全顔の box + 行灯疑いフラグ）
review:  detections.json ──(人間・対話)──→ work/decisions.json（疑い顔への y/n）
render:  in/ + 両JSON ──(無人・軽い)──→ out/
```

- CLI: `main.py [detect|review|render|all] -i in -o out -w work`。
  サブコマンド省略時は `all`（3 フェーズ連続実行）で従来の使い勝手を維持。
  Makefile の `uv run main.py --in_dir .. --out_dir ..` はそのまま動くこと。
- 再開可能性:
  - detect: detections.json に既にあるファイルはスキップ。バッチごとに JSON 保存（中断対応）。
  - review: 未回答の疑い顔だけ質問。1 回答ごとに JSON 保存。
  - render: 出力ファイルが既に存在すればスキップ（`--force` で再生成）。
    従来の「出力ディレクトリが空でなければ exit」は廃止。
- render 時に未回答の疑い顔が残っていたらエラーで中断（誤ってぼかす/ぼかさないを防ぐ）。
- YOLO のロードは detect 内に移動（review/render でモデルを読まない）。
- 押し間違いは decisions.json を手で直して該当出力を消す（または `--force`）→ render 再実行で修復。
- JSON 保存はテンポラリファイル経由のアトミック書き込みにする。

detections.json スキーマ:

```json
{
  "meta": {"model": "yolov11m-face.pt", "imgsz": 1280, "conf": 0.25},
  "files": {
    "rel/path.jpeg": {
      "faces": [{"box": [x1, y1, x2, y2], "suspect": true}]
    }
  }
}
```

decisions.json スキーマ（box は detections 側と突き合わせるキー）:

```json
{
  "rel/path.jpeg": [{"box": [x1, y1, x2, y2], "andon": true}]
}
```

### 2. EXIF の保持（piexif 追加）

- cv2.imread → cv2.imwrite は EXIF を全て落とす。render での書き出し後、piexif で
  元ファイルから EXIF をコピーする（JPEG のみ。PNG は対象外）。
- 注意 1: cv2.imread は EXIF Orientation を解釈してピクセルを回転させるため、
  コピー後に Orientation タグを 1 にリセットしないと二重回転する。
- 注意 2: **EXIF 内サムネイルには元画像（ぼかし前の顔）が残る**ため、必ず除去する
  （`exif_dict["thumbnail"] = None`）。プライバシー上必須。
- piexif.load / dump は壊れた EXIF で例外を出すことがあるので try/except で警告に留める。

### 3. Apple Silicon GPU (MPS) の利用

- ultralytics はデフォルト CPU。`torch.backends.mps.is_available()` なら
  `predict(device="mps")` を使う。`--device` で明示指定も可能に（デフォルト auto）。

### 4. 検出の knob（--imgsz / --conf）と小さい顔の検出漏れ対策

- YOLO デフォルトは 640px に縮小して推論。4000〜6000px の写真では遠くの顔が
  縮小後に数 px になり検出漏れする。**デフォルトを imgsz=1280 に引き上げる**
  （MPS 化とセットなら 1000 枚でも現実的な時間）。
- `--conf`（デフォルト 0.25）も公開。下げると拾い漏れ減・確認増。
- detections.json の meta に imgsz/conf を記録し、再開時に値が違ったら警告
  （混在検出を防ぐ。作業をやり直す場合は work dir を消す）。

### 5. 入力の堅牢性

- `rglob("*.jpg")` は `.JPG`/`.JPEG` を拾わない → Python 3.12 の
  `rglob(pattern, case_sensitive=False)` を使う。
- `cv2.imread` が None（壊れた画像）のとき predict でクラッシュ → 警告してスキップ。

### 6. ぼかしの品質

- 検出 box ちょうどのぼかしは輪郭（あご・額）が残るので box を 15% パディング
  （画像端でクランプ）。
- カーネル固定 (101,101) は巨大な顔に不十分 → 顔サイズに比例させる
  （`k = max(51, (max(w,h)//2) | 1)`、奇数化。2 パスは維持）。

### 7. レビュー UI の改善

- 4000〜6000px の画像をそのまま imshow すると画面からはみ出す →
  `cv2.namedWindow(WINDOW_NORMAL)` + 画面に収まるサイズに resizeWindow。
- 枠線・文字サイズを画像サイズに応じてスケール（現状 0.5 は高解像度で読めない）。
- ウィンドウをマウスで閉じると waitKey ループから抜けられない →
  `cv2.getWindowProperty(WND_PROP_VISIBLE)` を監視し、閉じたら進捗保存して正常終了。
- `q` キーで中断可能に（回答済み分は保存されているので再開できる）。
- 進捗表示（何件中何件目か）を画面に出す。

### 8. 出力品質（★品質はできるだけ下げない）

- cv2.imwrite の JPEG 品質はデフォルト 95 → **デフォルトを 97 に引き上げ**、
  `--jpeg_quality` で調整可能に。PNG は無劣化なのでそのまま。
- 再エンコードは 1 回だけ（in → out の 1 パス）なので世代劣化はしない。

### 9. ドキュメント・周辺

- README を新しい 3 フェーズの使い方・再開方法・decisions.json の直し方・
  EXIF の扱いに合わせて更新。
- Makefile: `WORK_DIR` パラメータ追加、`detect`/`review`/`render` ターゲット追加。
- .gitignore に `/work` を追加。

## コミット計画

1. `add improvement plan`（このファイル）
2. `split pipeline into detect/review/render phases`（改善 1）
3. `preserve exif metadata in jpeg outputs`（改善 2、piexif 依存追加）
4. `use apple silicon gpu (mps) when available`（改善 3）
5. `add imgsz/conf options and raise default imgsz to 1280`（改善 4）
6. `handle uppercase extensions and unreadable images`（改善 5）
7. `pad blur region and scale kernel with face size`（改善 6）
8. `improve review window ux`（改善 7）
9. `add jpeg quality option with higher default`（改善 8）
10. `update readme and makefile`（改善 9）

## 追加改善（実装中のフィードバックによる）

### 10. 上部の小さい顔も行灯疑いにする

- IMG_5984.jpeg / IMG_6257.jpeg で、画像上部 (y 中心 ≈ 0.3h) の小さい行灯顔
  (ratio ≈ 0.0025 < 0.004) が自動ぼかしされてしまった。
- 実際の人の顔はサンプル全 8 枚で y 中心 ≥ 0.65h に集中しているのに対し、
  行灯の顔は上部に来る → **顔中心が上位 40% (FACE_TOP_BAND=0.4) なら
  サイズによらず疑い扱い**にした。サンプルではレビュー追加はゼロで、
  該当の行灯顔 2 つだけが正しくレビュー対象になった。
- あわせて suspect 判定を detect 時ではなく review/render 時に計算するよう変更
  （detections.json には size と box のみ保存）。閾値調整でモデル再実行が不要に。

### 11. 楕円ぼかし

- 矩形ぼかしは不自然なので、境界をフェザリングした楕円マスクで合成する方式に変更。

## 実装で得られた知見

- **MPS の初回推論が一度だけ全顔ロストを返した**（再現せず）。安全のため
  detect の最初のバッチだけ CPU と件数を突き合わせ、不一致なら CPU に
  フォールバックするガードを入れた。顔の見逃しはこのツールの最悪の故障モード。
- **imgsz=1280 vs 640**: 昼間の群衆写真では 1280 が実在の顔を 4〜7 個多く拾う
  (640 では縮小で消える)。夜間写真では 640 の誤検出 (ねぶた模様など) が
  1280 で消える傾向。デフォルト 1280 が妥当。
- in/IMG_6257.jpeg は既にぼかし済みの出力画像なので検証データとしては特殊
  （ぼかし痕を顔として再検出する）。閾値調整の参考にする際は注意。

## 検証方法

- `in/` に 6 枚のサンプル画像あり（行灯写真を含む）。work/out はスクラッチディレクトリを
  使って `uv run main.py detect -i in -w <tmp>/work` → detections.json を目視確認。
- review は対話的なので自動検証しない。代わりに decisions.json を手書きして
  render を検証（andon=true の顔がぼけていないこと、他の顔がぼけていること）。
- EXIF: piexif で in/out の EXIF を比較。Orientation が 1 になっていること、
  thumbnail が消えていること、撮影日時などが保持されていることを確認。
- 各コミット前に `make format lint`。
