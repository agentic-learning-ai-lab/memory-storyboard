import tqdm
import os
import multiprocessing as mp

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Root directory of video dataset.')
    parser.add_argument('--frames_dir', help='Root directory to place frames.')
    parser.add_argument('--workers', default=2, type=int, help='Number of parallel workers.')
    parser.add_argument('--sta_idx', default=0, type=int, help='Video Start Index.')
    parser.add_argument('--end_idx', default=500, type=int, help='Video End Index (total 436 for K-Cam).')
    args = parser.parse_args()

    videos_fns = [f"{root}/{fn}" for root, subdir, files in os.walk(args.video_dir)
                  for fn in files if fn.endswith('.mp4') or fn.endswith('.avi')]
    videos_fns = sorted(videos_fns)
    print(len(videos_fns))
    videos_fns = videos_fns[args.sta_idx:args.end_idx]
    jobs = []
    for vfn in videos_fns:
        basename = os.path.basename(vfn).split('.')[0]
        dst_fns = f'{args.frames_dir}/{basename}/%06d.jpg'
        os.makedirs(os.path.dirname(dst_fns), exist_ok=True)
        scale = '-vf scale="-2:\'min(256,ih)\'"'
        command = f'ffmpeg -y -i {vfn} {scale} -start_number 0 -r 10 {dst_fns}'
        jobs += [command]

    pool = mp.Pool(args.workers)
    for _ in tqdm.tqdm(pool.imap_unordered(os.system, jobs), total=len(jobs)):
        pass

if __name__ == '__main__':
    main()
