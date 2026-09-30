"""절단된 stage 데이터셋 검증: 에피소드 길이/분할을 manifest와 대조하고 시작·중간·끝 프레임 미리보기 PNG를 저장한다.

Usage:
    uv run python custom_scripts/check_stage_split.py                # 기본: unfold, 앞 6개 에피소드 미리보기
    uv run python custom_scripts/check_stage_split.py --episodes 0 30 99
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/home/youngchan/.cache/huggingface/lerobot/Woohi123")
CAMERA = "observation.images.astra_rgb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="unfold")
    parser.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=Path("outputs/stage_split_preview.png"))
    args = parser.parse_args()

    root = ROOT / f"towel_fold_v1_balanced150_stage_{args.stage}"
    manifest = json.loads((root / "stage_split_manifest.json").read_text())
    ds = LeRobotDataset(f"Woohi123/towel_fold_v1_balanced150_stage_{args.stage}", root=root, return_uint8=True, revision="main")
    done = ds.meta.total_episodes
    print(f"완료 여부: {manifest['complete']} | 저장된 에피소드 {done}/{len(manifest['samples'])} | 프레임 {ds.meta.total_frames}")

    bad = 0
    for i in range(done):
        s = manifest["samples"][i]
        expected = s["end_offset"] - s["start_offset"]
        actual = ds.meta.episodes[i]["length"]
        ok = expected == actual
        bad += not ok
        print(f"ep{i:03d} src={s['source_episode']:3d} {s['split']:5s} 기대={expected:5d} 실제={actual:5d} {'OK' if ok else 'MISMATCH'}")
    sources = {"train": set(), "eval": set()}
    for s in manifest["samples"][:done]:
        sources[s["split"]].add(s["source_episode"])
    print(f"train/eval 원본 에피소드 겹침: {sorted(sources['train'] & sources['eval']) or '없음'}")
    print(f"길이 불일치: {bad}건")

    rows = []
    for ep in [e for e in args.episodes if e < done]:
        meta = ds.meta.episodes[ep]
        start, n = meta["dataset_from_index"], meta["length"]
        frames = [np.asarray(ds[start + k][CAMERA]) for k in (0, n // 2, n - 1)]
        frames = [(f.transpose(1, 2, 0) if f.shape[0] == 3 else f) for f in frames]
        frames = [f if f.dtype == np.uint8 else (f * 255).astype(np.uint8) for f in frames]
        row = Image.fromarray(np.concatenate(frames, axis=1))
        ImageDraw.Draw(row).text((5, 5), f"ep{ep} src={manifest['samples'][ep]['source_episode']} (시작 | 중간 | 끝)", fill=(255, 255, 0))
        rows.append(np.asarray(row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"미리보기 저장: {args.out}")


if __name__ == "__main__":
    main()
