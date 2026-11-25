"""Controlled geometry checks, independent of the real photographs' unknown labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from stamp_hole_inspect import (Detection, Geometry, analyze_lines, line_members,
                                predicted_slots, warp_design)


def validate():
    geometry = Geometry()
    horizontal = [np.polynomial.Polynomial([600 + row*500]) for row in range(6)]
    vertical = [np.polynomial.Polynomial([300 + col*380]) for col in range(11)]
    ids, ideal = predicted_slots([horizontal,vertical],geometry)
    x, y = (ideal[:,0]-2200)/2200, (ideal[:,1]-2000)/2000
    good_xy = ideal + np.column_stack([12*x*y*y, 15*y*x*x])
    x, y = (good_xy[:,0]-2200)/2200, (good_xy[:,1]-2000)/2000
    bad_xy = good_xy + np.column_stack([35+5*y+9*x*x*y, 50-8*x+12*x*y*y])
    def detection(xy):
        observations = np.column_stack([xy,np.full(len(xy),.99),np.full(len(xy),100),np.zeros(len(xy))])
        return Detection(ids,xy.copy(),observations,np.ones(len(xy),bool),np.empty((0,4)),{})
    good = detection(good_xy)
    clean_lines, clean_records, _ = analyze_lines(detection(bad_xy),good,geometry)
    max_clean = max(abs(r['registered_reference_normal_px']) for r in clean_records if r['reference_pair_reliable'])
    assert max_clean < .02, max_clean
    target = ids.index(('V',5,11))
    shifted = bad_xy.copy();shifted[target,0] += 6
    lines, records, _ = analyze_lines(detection(shifted),good,geometry)
    hole = next(r for r in records if r['line']=='V6' and r['pitch_index']==11)
    assert abs(hole['bad_normal_to_curve_px']-6) < .20, hole
    assert abs(hole['registered_reference_normal_px']-6) < .05, hole
    members = line_members(ids,geometry)['V6']
    shifted = bad_xy.copy();shifted[[i for _,i in members],0] += 6
    lines, _, _ = analyze_lines(detection(shifted),good,geometry)
    moved_line = next(r for r in lines if r['line']=='V6')
    assert abs(moved_line['median_reference_normal_px']-6) < .05, moved_line
    assert moved_line['max_abs_hole_to_curve_px'] < .1, moved_line
    # A missing hole and a shifted provisional index must not produce a one-pitch defect.
    bad = detection(bad_xy.copy());bad.accepted[target] = False
    bad.observations[[target+1,target+2]] = bad.observations[[target+2,target+1]]
    _, records, _ = analyze_lines(bad,good,geometry)
    local = [r for r in records if r['line']=='V6' and r['reference_pair_reliable']]
    assert max(abs(r['registered_reference_along_px']) for r in local) < .03
    return dict(passed=True,synthetic_only=True,
                smooth_distortion_max_false_normal_px=max_clean,
                injected_single_hole_px=6,measured_single_hole_curve_px=hole['bad_normal_to_curve_px'],
                injected_whole_line_px=6,measured_whole_line_px=moved_line['median_reference_normal_px'],
                missing_hole_and_index_swap='passed',
                limitation='These controls validate geometry behavior, not real-photo recall or camera calibration.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = validate()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
