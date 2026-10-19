import sys,json,time,hashlib
from pathlib import Path
import cv2
sys.path.insert(0,str(Path('cpp-core/tools').resolve()))
from coarse_experiment import CoarsePattern
from real_image_experiments import Library,image_view,scaled_template
cv2.setNumThreads(1)
lib=Library(str(Path('build/pipeline-baseline/libfpm_c.dylib').resolve()))
cases=[('src7','Test Images/Src7.bmp','Test Images/Dst7.bmp',[1.],180,.5),('controlled6','build/coarse-ncc/inputs/controlled.png','Test Images/Dst3.bmp',[.75,1.,1.25],180,.5),('heldout12','build/coarse-ncc/inputs/heldout.png','Test Images/Dst3.bmp',[.75,1.,1.25],180,.5),('8k_t256','build/coarse-ncc/inputs/8k_t256.png','build/coarse-ncc/inputs/template256.png',[1.],0,.7)]
rows=[]
for name,src,tmp,scales,angle,score in cases:
 source=cv2.imread(src,0);template=cv2.imread(tmp,0);cfg=lib.config();cfg.max_targets=70 if angle else 12;cfg.min_score=score;cfg.max_overlap=.2 if angle else 0.;cfg.tolerance_angle=angle;cfg.stop_layer=1;cfg.subpixel=0
 for backend in ('cpu','metal'):
  templates=[scaled_template(template,s) for s in scales]
  patterns=[CoarsePattern('build/fourier-gpu',image_view(t),256,backend=backend) for t in templates]
  old=[lib.learn(image_view(t),256) for t in templates]
  expected=[p.match(image_view(source),cfg) for p in old]
  def run():
   r=[]
   for p,ref in zip(patterns,expected):
    matches=p.match(image_view(source),cfg,profile=True)
    assert matches==ref,(name,backend)
    r.append({'profile':p.last_profile,'metrics':p.last_metrics})
   return r
  first=run();start=time.perf_counter()
  while time.perf_counter()-start<.5:run()
  measured=[run() for _ in range(3)]
  row={'case':name,'backend':backend,'first':first,'measured':measured,'exact':True};rows.append(row)
  rep=sorted(measured,key=lambda calls:sum(r['profile']['core_wall_ms'] for r in calls))[1]
  print(name,backend,{k:round(sum(r['profile'][k] for r in rep),3) for k in rep[0]['profile'] if k.endswith('_ms')},flush=True)
  for p in patterns+old:p.close()
Path('build/pipeline-stages/decision-probe.json').write_text(json.dumps({'purpose':'Early profiling decision probe, not formal speed acceptance','binary_sha256':hashlib.sha256(Path('build/fourier-gpu/experiments/libfpm_coarse_experiment.dylib').read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
