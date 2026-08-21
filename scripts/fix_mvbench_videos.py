#!/usr/bin/env python3
"""MVBench 비디오 손상 진단 및 복구.

증상: 평가 중 "RuntimeError: Error reading .../mvbench_video/....mp4"
원인: 디스크 부족 등으로 압축 해제된 mp4가 잘림 (원본 zip은 HF 캐시에 그대로 있음)

진단만:  python scripts/fix_mvbench_videos.py
복구까지: python scripts/fix_mvbench_videos.py --fix
특정 파일: python scripts/fix_mvbench_videos.py --file ZS0XR.mp4 --fix
"""

import argparse
import os
import shutil
import zipfile

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MIN_OK = 10 * 1024  # 10KB 미만이면 잘린 것으로 간주


def find_video_root():
    for root, dirs, _ in os.walk(HF_HOME):
        for d in dirs:
            if d == "mvbench_video":
                return os.path.join(root, d)
    return None


def find_zips():
    zips = []
    for root, _, files in os.walk(os.path.join(HF_HOME, "hub")):
        if "MVBench" not in root:
            continue
        zips += [os.path.join(root, f) for f in files if f.endswith(".zip")]
    return zips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="손상 파일을 zip에서 다시 풀기")
    ap.add_argument("--file", default=None, help="특정 파일명만 점검/복구 (예: ZS0XR.mp4)")
    ap.add_argument("--verify", action="store_true",
                    help="decord로 실제로 열어보며 검사 (크기는 정상인데 내용이 깨진 경우 탐지, 수 분 소요)")
    args = ap.parse_args()

    # 1) 디스크
    if not os.path.isdir(HF_HOME):
        print(f"[경고] HF_HOME 경로가 없습니다: {HF_HOME}")
        print("  echo $HF_HOME 로 확인하고, 비어 있으면 source ~/.bashrc 후 다시 실행하세요.")
        return
    total, used, free = shutil.disk_usage(HF_HOME)
    gb = 1024 ** 3
    print(f"[디스크] {HF_HOME}: 여유 {free/gb:.1f}GB / 전체 {total/gb:.1f}GB ({used/total*100:.0f}% 사용)")
    if free < 5 * gb:
        print("  ★ 여유 공간이 5GB 미만입니다 — 이것이 손상의 원인일 가능성이 높습니다.")
        print("    공간 확보 예: rm -rf ~/videomme_videos  (원본 zip은 캐시에 남아 있어 언제든 재추출 가능)")

    # 2) 압축 해제 폴더
    root = find_video_root()
    if not root:
        print("[비디오] mvbench_video 폴더를 찾지 못했습니다 — 평가 첫 실행 시 자동 생성됩니다.")
        return
    print(f"[비디오] {root}")

    mp4s = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm", ".mov")):
                mp4s.append(os.path.join(r, f))
    print(f"  파일 {len(mp4s)}개")

    if args.file:
        targets = [p for p in mp4s if os.path.basename(p) == args.file]
        if not targets:
            print(f"  '{args.file}' 이(가) 폴더에 없습니다 — zip에서 새로 풀어야 합니다.")
            broken_names = {args.file}
        else:
            for p in targets:
                print(f"  {args.file}: {os.path.getsize(p)} bytes")
            broken_names = {args.file}
    elif args.verify:
        try:
            from decord import VideoReader, cpu
        except ImportError:
            print("  ★ decord가 없습니다. llava 또는 internvl 환경에서 실행하세요.")
            return
        broken = []
        for i, p in enumerate(mp4s):
            try:
                vr = VideoReader(p, ctx=cpu(0), num_threads=1)
                if len(vr) < 1:
                    raise RuntimeError("frame count 0")
                vr[0]  # 첫 프레임 실제 디코딩
            except Exception as e:
                broken.append(p)
                print(f"    [깨짐] {os.path.relpath(p, root)}: {type(e).__name__}")
            if (i + 1) % 500 == 0:
                print(f"  검사 {i+1}/{len(mp4s)} — 지금까지 손상 {len(broken)}개", flush=True)
        print(f"  디코딩 검사 결과: 손상 {len(broken)}개 / 전체 {len(mp4s)}개")
        if not broken:
            print("  → 모든 파일이 정상적으로 열립니다. 원인이 파일 손상이 아닐 수 있습니다.")
            return
        broken_names = {os.path.basename(p) for p in broken}
    else:
        broken = [p for p in mp4s if os.path.getsize(p) < MIN_OK]
        print(f"  손상 의심(10KB 미만): {len(broken)}개")
        for p in broken[:10]:
            print(f"    {os.path.getsize(p):>8} bytes  {os.path.relpath(p, root)}")
        if len(broken) > 10:
            print(f"    ... 외 {len(broken)-10}개")
        if not broken:
            print("  → 크기 기준으로는 이상 없음. 특정 파일이 문제면 --file <파일명>으로 점검하세요.")
            return
        broken_names = {os.path.basename(p) for p in broken}

    # 3) zip에서 복구
    zips = find_zips()
    print(f"[원본] MVBench zip {len(zips)}개 발견")
    if not zips:
        print("  ★ 원본 zip이 없습니다. 데이터셋 재다운로드 필요:")
        print("    hf download OpenGVLab/MVBench --repo-type dataset")
        return

    plan = []  # (zip, member, dest)
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if os.path.basename(m) in broken_names:
                        plan.append((z, m))
        except zipfile.BadZipFile:
            print(f"  [warn] 손상된 zip: {z}")

    print(f"[복구 대상] zip 안에서 {len(plan)}개 매칭")
    if not plan:
        print("  ★ zip에서 해당 파일을 찾지 못했습니다. 데이터셋을 다시 받는 편이 빠릅니다:")
        print("    hf download OpenGVLab/MVBench --repo-type dataset")
        return

    if not args.fix:
        print("\n복구하려면 --fix 를 붙여 다시 실행하세요:")
        print("  python scripts/fix_mvbench_videos.py --fix")
        return

    done = 0
    for z, m in plan:
        with zipfile.ZipFile(z) as zf:
            zf.extract(m, root)
        out = os.path.join(root, m)
        print(f"  복구: {m} ({os.path.getsize(out)} bytes)")
        done += 1
    print(f"\n{done}개 복구 완료 — 평가를 다시 실행하세요.")


if __name__ == "__main__":
    main()
