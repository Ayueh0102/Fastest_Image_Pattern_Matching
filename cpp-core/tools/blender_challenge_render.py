"""Run with Blender --background --python this_file -- --output DIR.

CPU Cycles photometric/antialiasing control, orthographic frontal planar poses.
Raw RGB render, complete packed .blend and independent camera-projected corners
are saved. This is a renderer-domain validation set, not real camera footage.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
WIDTH=HEIGHT=512
TW,TH=96,72


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(changed_light=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=32
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_denoising=False
    scene.cycles.seed=341916
    scene.cycles.max_bounces=2
    scene.cycles.diffuse_bounces=1
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=WIDTH;scene.render.resolution_y=HEIGHT;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
    scene.render.film_transparent=False
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    scene.world=bpy.data.worlds.new('Neutral environment')
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.13,.13,.13,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3
    camera_data=bpy.data.cameras.new('Orthographic camera')
    camera=bpy.data.objects.new('Orthographic camera',camera_data)
    scene.collection.objects.link(camera)
    camera.location=(0,0,1000);camera.rotation_euler=(0,0,0)
    camera.data.type='ORTHO';camera.data.ortho_scale=512;camera.data.lens=50
    camera.data.clip_start=.1;camera.data.clip_end=2000;scene.camera=camera
    light_data=bpy.data.lights.new('Area illumination','AREA')
    light=bpy.data.objects.new('Area illumination',light_data);scene.collection.objects.link(light)
    light.location=(-260,180,550) if changed_light else (0,0,700)
    light.rotation_euler=(0,0,0)
    light.data.energy=350000 if changed_light else 500000
    light.data.shape='DISK';light.data.size=300 if changed_light else 900
    light.data.color=(1.,.65,.35) if changed_light else (1.,1.,1.)
    bg=bpy.data.materials.new('Matte background');bg.use_nodes=True
    bsdf=bg.node_tree.nodes['Principled BSDF'];bsdf.inputs['Base Color'].default_value=(.28,.28,.28,1)
    bsdf.inputs['Roughness'].default_value=1;bsdf.inputs['Specular IOR Level'].default_value=0
    plane('Background',512,512,[256,256],0,1,bg,z=-2)
    return scene


def material(texture_path,altered=False):
    material=bpy.data.materials.new('Altered local identity' if altered else 'Photographed target')
    material.use_nodes=True
    nodes=material.node_tree.nodes;bsdf=nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value=1;bsdf.inputs['Specular IOR Level'].default_value=0
    texture=nodes.new('ShaderNodeTexImage')
    texture.image=bpy.data.images.load(str(texture_path),check_existing=False)
    texture.image.pack();texture.interpolation='Linear';texture.extension='EXTEND'
    material.node_tree.links.new(texture.outputs['Color'],bsdf.inputs['Base Color'])
    return material


def plane(name,width,height,center,angle,scale,mat,z=0):
    bpy.ops.mesh.primitive_plane_add(size=2,location=(center[0]-256,256-center[1],z))
    obj=bpy.context.object;obj.name=name
    obj.scale=(width*scale/2,height*scale/2,1)
    obj.rotation_euler.z=math.radians(angle)
    obj.data.materials.append(mat)
    return obj


def truth_for(scene,obj,index,center,angle,scale,identity,occluded_fraction):
    # World-to-camera projection is independent of the hand-written 2D rotation
    # formula in the outer checker. Local corner order uses image-space y down.
    bpy.context.view_layer.update()
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    corners=[]
    for corner in ((-1,1,0),(1,1,0),(1,-1,0),(-1,-1,0)):
        p=world_to_camera_view(scene,scene.camera,evaluated.matrix_world@Vector(corner))
        corners.append([p.x*WIDTH,(1-p.y)*HEIGHT])
    p=world_to_camera_view(scene,scene.camera,evaluated.matrix_world.translation)
    return dict(index=index,identity=identity,pos_x=center[0],pos_y=center[1],angle=angle,scale=scale,
        corners=corners,projected_center=[p.x*WIDTH,(1-p.y)*HEIGHT],
        projected_corner_source='Blender evaluated object matrix + world_to_camera_view',
        nominal_strip_occlusion_fraction=occluded_fraction,object_name=obj.name)


def poses(seed):
    rng=random.Random(seed)
    return [dict(center=[(index%3+.5)*512/3+rng.uniform(-2,2),(index//3+.5)*256+rng.uniform(-2,2)],
                 angle=rng.uniform(-150,150),scale=rng.uniform(.88,1.12)) for index in range(6)]


def render_one(output,name,texture,altered_texture,items,changed_light=False,occlusion=False,distractors=False):
    scene=setup(changed_light)
    target=material(texture);altered=material(altered_texture,True)
    occluder=bpy.data.materials.new('Opaque matte occluder');occluder.use_nodes=True
    bsdf=occluder.node_tree.nodes['Principled BSDF'];bsdf.inputs['Base Color'].default_value=(.2,.24,.3,1)
    bsdf.inputs['Roughness'].default_value=1;bsdf.inputs['Specular IOR Level'].default_value=0
    objects=[]
    for index,item in enumerate(items):
        center,angle,scale=item['center'],item['angle'],item['scale']
        identity='altered_patch' if distractors and index in (1,4) else 'target'
        fraction=(0.,.1,.2,.3,.4,.5)[index] if occlusion else 0.
        obj=plane('Target_%02d'%index,TW,TH,center,angle,scale,altered if identity!='target' else target)
        objects.append(truth_for(scene,obj,index,center,angle,scale,identity,fraction))
        if fraction:
            radians=math.radians(angle)
            local_x=TW*(fraction-1)/2*scale
            # Cover the left-hand local strip at z=.1, without camera parallax.
            strip_center=[center[0]+math.cos(radians)*local_x,center[1]-math.sin(radians)*local_x]
            plane('Occluder_%02d'%index,TW*fraction,TH,strip_center,angle,scale,occluder,z=.1)
    scene.render.filepath=str(output/(name+'-rgb.png'))
    bpy.ops.wm.save_as_mainfile(filepath=str(output/(name+'.blend')))
    bpy.ops.render.render(write_still=True)
    metadata=dict(name=name,objects=objects,truth=[o for o in objects if o['identity']=='target'],
        distractors=[o for o in objects if o['identity']!='target'],changed_light=changed_light,
        occlusion=occlusion,template_plane_pixels=[TW,TH],source_shape=[HEIGHT,WIDTH],
        render_engine='Cycles CPU',samples=32,adaptive_sampling=False,denoising=False,
        color_management='Standard / None / exposure0 / gamma1',projection='orthographic frontal',
        perspective_out_of_model_excluded=True,blender_version=bpy.app.version_string,
        blend_sha256=sha(output/(name+'.blend')),rgb_sha256=sha(output/(name+'-rgb.png')))
    save(output/(name+'.json'),metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--texture',type=Path,required=True)
    parser.add_argument('--altered-texture',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite rendered evidence')
    out.mkdir(parents=True,exist_ok=True)
    render_one(out,'reference',args.texture,args.altered_texture,
        [dict(center=[256.,256.],angle=0.,scale=1.)])
    development=poses(2026103411);heldout=poses(2026103412)
    for name,items,flags in (
        ('blender_neutral',development,{}),
        ('blender_changed_light',heldout,dict(changed_light=True)),
        ('blender_occlusion',heldout,dict(occlusion=True)),
        ('blender_distractors',heldout,dict(distractors=True))):
        render_one(out,name,args.texture,args.altered_texture,items,**flags)
    save(out/'generation.json',dict(script_sha256=sha(__file__),texture_sha256=sha(args.texture),
        altered_texture_sha256=sha(args.altered_texture),seeds=[2026103411,2026103412],
        crop_xywh=[208,220,96,72],independent_pose_sets=2,independent_geometric_poses=12,
        note='Heldout conditions reuse one six-pose geometry; not eighteen independent new poses.'))

if __name__=='__main__':main()
