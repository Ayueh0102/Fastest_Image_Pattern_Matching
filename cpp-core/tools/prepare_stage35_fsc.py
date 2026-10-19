"""Fetch a small, pinned FSC147 research subset without changing its RGB pixels.

Images remain local: the upstream code license does not establish photo rights.
The annotation mirror must match the author's Git blob exactly. Counts are
category point labels, NOT same-template identities or calibrated poses.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np

REVISION = '3e420cb6537e803dd6d4516623ce82a79c0317b8'
ANNOTATION_BLOB = 'bc97805951dab999aca083b4de1e5a28de940213'
BASE = f'https://huggingface.co/datasets/isentropic/FSC147/resolve/{REVISION}/'
IDS = ['2927', '2928', '2931', '2933', '7222', '7417', '7263',
       '825', '1913', '1921', '1924', '1944']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dest = args.output.resolve()
    raw = fetch(BASE+'annotation_FSC147_384.json', dest/'cache/annotation.json')
    git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if git_blob != ANNOTATION_BLOB:
        raise ValueError('Annotation mirror does not match the official Git blob')
    annotations = json.loads(raw)

    def one(identifier):
        name = identifier+'.jpg'
        url = BASE+'images_384_VarV2/'+name
        path = dest/'images'/name
        data = fetch(url, path)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode '+name)
        annotation = annotations[name]
        h, w = image.shape[:2]
        points = np.asarray(annotation['points'], float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid points')
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            raise ValueError('Point outside image '+name)
        return dict(id=identifier, image=str(path), image_sha256=sha(data), url=url,
                    shape=list(image.shape), annotation=annotation,
                    category='stamps' if identifier in IDS[:7] else 'bottle caps',
                    official_point_count=len(points))

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, IDS))
    report = dict(dataset='FSC147', mirror_revision=REVISION,
                  official_repository='https://github.com/cvlab-stonybrook/LearningToCountEverything',
                  annotation_git_blob=git_blob, annotation_sha256=sha(raw),
                  annotation_mirror_matches_official=True, images=rows,
                  scope='Local research; no image redistribution license inferred from code license.',
                  labels='Category centers and three exemplar boxes only; no instance identity, color anomaly or pose GT.')
    (dest/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    # QA contact sheet only; never used as an algorithm input.
    cells=[]
    for row in rows:
        im=cv2.imread(row['image']); scale=min(350/im.shape[1], 280/im.shape[0])
        im=cv2.resize(im,None,fx=scale,fy=scale)
        tile=np.full((320,370,3),245,np.uint8)
        tile[30:30+im.shape[0],:im.shape[1]]=im
        cv2.putText(tile, f"{row['id']} / {row['official_point_count']} points",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cells.append(tile)
    cv2.imwrite(str(dest/'inspection.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,len(cells),3)]))
    print(json.dumps([dict(id=r['id'], count=r['official_point_count'], shape=r['shape']) for r in rows]))


if __name__ == '__main__':
    main()
