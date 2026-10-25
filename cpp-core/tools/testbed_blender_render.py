"""Test-bed v1 Blender scenes.

Run from the repository root after `testbed_generate.py textures`:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
    --python cpp-core/tools/testbed_blender_render.py -- --textures DIR --output DIR [--only a,b]
    [--seed N --rgb-order-seed N --rgb-layout-seed N]   (defaults: the test-bed v1 seeds)

Cycles CPU with a fixed seed and thread count, Standard view transform, 8-bit PNG.
Every object is a textured plane. Ground-truth corners and centres come from
Blender's evaluated object matrices and world_to_camera_view, independently of
the 2D pose formula; frontal scenes are checked against that formula later.
Tilted and perspective scenes are outside the matcher's similarity model and
are labelled as such; their GT is the projected quadrilateral.
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

GRAY_W, GRAY_H = 140, 102   # Test Images/Dst3.bmp
RGB_W, RGB_H = 128, 80      # testbed_generate.capsule_texture
SAMPLES, SEED, THREADS = 48, 2026102410, 8
RGB_ORDER_SEED, RGB_LAYOUT_SEED = 2026102416, 2026102417
WARM, COOL = (1., .70, .42), (.75, .85, 1.)
# Measured on a 768^2 smoke render: 500000 W per (512 px)^2 gave white -> 0.10 linear.
# This gain maps a white Lambertian surface under the neutral light to about 0.9 linear.
EXPOSURE_GAIN = 8.8


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Scene:
    def __init__(self, width, height, *, perspective_deg=0., lights=('neutral',)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.width, self.height = width, height
        s = self.scene = bpy.context.scene
        s.render.engine = 'CYCLES'; s.cycles.device = 'CPU'; s.cycles.samples = SAMPLES
        s.cycles.use_adaptive_sampling = False; s.cycles.use_denoising = False; s.cycles.seed = SEED
        s.cycles.max_bounces = 3; s.cycles.diffuse_bounces = 1; s.cycles.glossy_bounces = 1
        s.render.threads_mode = 'FIXED'; s.render.threads = THREADS
        s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = width, height, 100
        s.render.image_settings.file_format = 'PNG'; s.render.image_settings.color_mode = 'RGB'
        s.render.image_settings.color_depth = '8'; s.render.film_transparent = False
        s.view_settings.view_transform = 'Standard'; s.view_settings.look = 'None'
        s.view_settings.exposure = 0; s.view_settings.gamma = 1
        s.world = bpy.data.worlds.new('Neutral environment'); s.world.use_nodes = True
        bg = s.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (.13, .13, .13, 1); bg.inputs['Strength'].default_value = .3
        cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
        s.collection.objects.link(cam); s.camera = cam
        side = max(width, height)
        if perspective_deg:
            cam.data.type = 'PERSP'; cam.data.lens = 50; cam.data.sensor_width = 36
            distance = 1.75 * side
            t = math.radians(perspective_deg)
            cam.location = (0, -distance * math.sin(t), distance * math.cos(t)); cam.rotation_euler = (t, 0, 0)
            self.projection = f'perspective camera, {perspective_deg:g} deg off the plane normal'
        else:
            cam.data.type = 'ORTHO'; cam.data.ortho_scale = side
            cam.location = (0, 0, 2 * side); cam.rotation_euler = (0, 0, 0)
            self.projection = 'orthographic frontal'
        cam.data.clip_start = .1; cam.data.clip_end = 10 * side
        for kind in lights:
            self.light(kind)
        self.matte = self.flat_material('Matte background', (.28, .28, .28))
        self.plane('Background', width * 1.2, height * 1.2, [width / 2, height / 2], 0, 1, self.matte, z=-2)
        self.objects = []

    def light(self, kind):
        side = max(self.width, self.height)
        data = bpy.data.lights.new(kind, 'AREA'); data.shape = 'DISK'
        obj = bpy.data.objects.new(kind, data); self.scene.collection.objects.link(obj)
        scale = EXPOSURE_GAIN * (side / 512) ** 2
        if kind in ('neutral', 'warm'):
            obj.location = (0, 0, 1.37 * side); data.size = 1.76 * side; data.energy = 500000 * scale
            data.color = WARM if kind == 'warm' else (1., 1., 1.)
        elif kind in ('left_warm', 'right_cool'):
            sign = -1 if kind == 'left_warm' else 1
            obj.location = (sign * .55 * side, 0, .9 * side); data.size = .9 * side; data.energy = 260000 * scale
            obj.rotation_euler = (0, math.radians(sign * 28), 0)
            data.color = WARM if kind == 'left_warm' else COOL
        elif kind == 'glossy_key':
            obj.location = (-.45 * side, .35 * side, .55 * side); data.size = .08 * side; data.energy = 260000 * scale
            obj.rotation_euler = (math.radians(-30), math.radians(-38), 0)
        else:
            raise ValueError(kind)

    @staticmethod
    def flat_material(name, colour, roughness=1., specular=0.):
        m = bpy.data.materials.new(name); m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = (*colour, 1)
        b.inputs['Roughness'].default_value = roughness; b.inputs['Specular IOR Level'].default_value = specular
        return m

    @staticmethod
    def texture_material(path, roughness=1., specular=0.):
        m = bpy.data.materials.new(Path(path).stem); m.use_nodes = True
        nodes = m.node_tree.nodes; b = nodes['Principled BSDF']
        b.inputs['Roughness'].default_value = roughness; b.inputs['Specular IOR Level'].default_value = specular
        t = nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(str(path), check_existing=True)
        t.image.pack(); t.interpolation = 'Linear'; t.extension = 'EXTEND'
        m.node_tree.links.new(t.outputs['Color'], b.inputs['Base Color'])
        return m

    def plane(self, name, width, height, centre, angle, scale, material, z=0., tilt=0.):
        bpy.ops.mesh.primitive_plane_add(size=2, location=(centre[0] - self.width / 2, self.height / 2 - centre[1], z))
        obj = bpy.context.object; obj.name = name
        obj.scale = (width * scale / 2, height * scale / 2, 1)
        obj.rotation_euler = (math.radians(tilt), 0, math.radians(angle))
        obj.data.materials.append(material)
        return obj

    def box(self, name, centre, size, z, material):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(centre[0] - self.width / 2, self.height / 2 - centre[1], z))
        obj = bpy.context.object; obj.name = name; obj.scale = size
        obj.data.materials.append(material)
        return obj

    def add_object(self, index, texture_w, texture_h, material, pose, **labels):
        obj = self.plane(f'Object_{index:02d}', texture_w, texture_h, pose['centre'], pose['angle'], pose['scale'],
                         material, tilt=pose.get('tilt', 0.))
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        corners = []
        for corner in ((-1, 1, 0), (1, 1, 0), (1, -1, 0), (-1, -1, 0)):
            p = world_to_camera_view(self.scene, self.scene.camera, evaluated.matrix_world @ Vector(corner))
            corners.append([p.x * self.width, (1 - p.y) * self.height])
        p = world_to_camera_view(self.scene, self.scene.camera, evaluated.matrix_world.translation)
        self.objects.append(dict(index=index, pos_x=pose['centre'][0], pos_y=pose['centre'][1], angle=pose['angle'],
            scale=pose['scale'], tilt_deg=pose.get('tilt', 0.), corners=corners,
            projected_center=[p.x * self.width, (1 - p.y) * self.height], object_name=obj.name, **labels))

    def render(self, output, name, *, template_plane, crop=None, notes=''):
        image = output / f'{name}.png'
        self.scene.render.filepath = str(image)
        bpy.ops.wm.save_as_mainfile(filepath=str(output / f'{name}.blend'))
        bpy.ops.render.render(write_still=True)
        save(output / f'{name}.json', dict(name=name, image=image.name, image_sha256=sha(image),
            blend_sha256=sha(output / f'{name}.blend'), source_shape=[self.height, self.width],
            template_plane_pixels=list(template_plane), template_crop_xywh=crop, projection=self.projection,
            render='Cycles CPU', samples=SAMPLES, seed=SEED, threads=THREADS, denoising=False,
            color_management='Standard / None / exposure 0 / gamma 1', blender_version=bpy.app.version_string,
            objects=self.objects, notes=notes))


def grid_poses(seed, columns, rows, cell, *, scale_range=(1., 1.), jitter=6., margin=0.):
    rng = random.Random(seed)
    out = []
    for index in range(columns * rows):
        row, column = divmod(index, columns)
        lo, hi = scale_range
        out.append(dict(centre=[(column + .5) * cell + rng.uniform(-jitter, jitter), margin + (row + .5) * cell + rng.uniform(-jitter, jitter)],
                        angle=rng.uniform(-180, 180), scale=math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    return out


def reference(output, name, texture, w, h, size, lights=('neutral',)):
    s = Scene(size[0], size[1], lights=lights)
    s.add_object(0, w, h, s.texture_material(texture), dict(centre=[size[0] / 2, size[1] / 2], angle=0., scale=1.), identity='target')
    s.render(output, name, template_plane=(w, h), crop=[size[0] // 2 - w // 2, size[1] // 2 - h // 2, w, h],
             notes='Template source: upright, scale 1, centred; crop is the exact plane rectangle.')


def build(textures, output, only, rgb_order_seed=RGB_ORDER_SEED, rgb_layout_seed=RGB_LAYOUT_SEED):
    meta = json.loads((textures / 'textures.json').read_text())
    gray, symmetric = textures / 'gray_dst3.png', textures / 'gray_dst3_symmetric180.png'
    jobs = {}

    def job(name):
        def wrap(fn):
            jobs[name] = fn
            return fn
        return wrap

    @job('bl_gray_reference')
    def _():
        reference(output, 'bl_gray_reference', gray, GRAY_W, GRAY_H, (768, 768))

    @job('bl_gray_crowded')
    def _():
        s = Scene(1536, 1536); m = s.texture_material(gray)
        for i, pose in enumerate(grid_poses(2026102411, 6, 6, 256, scale_range=(.8, 1.25))):
            s.add_object(i, GRAY_W, GRAY_H, m, pose, identity='target')
        s.render(output, 'bl_gray_crowded', template_plane=(GRAY_W, GRAY_H), notes='36 targets, scale 0.8-1.25, any angle.')

    @job('bl_gray_tilt')
    def _():
        s = Scene(768, 768); m = s.texture_material(gray)
        for i, pose in enumerate(grid_poses(2026102412, 3, 3, 256)):
            pose['tilt'] = 5. * i
            s.add_object(i, GRAY_W, GRAY_H, m, pose, identity='target')
        s.projection = 'orthographic, planes tilted 0-40 deg about their local x axis (outside similarity model)'
        s.render(output, 'bl_gray_tilt', template_plane=(GRAY_W, GRAY_H), notes='Tilt i*5 deg for object i.')

    @job('bl_gray_perspective')
    def _():
        s = Scene(768, 768, perspective_deg=25.); m = s.texture_material(gray)
        for i, pose in enumerate(grid_poses(2026102413, 3, 3, 256)):
            s.add_object(i, GRAY_W, GRAY_H, m, pose, identity='target')
        s.render(output, 'bl_gray_perspective', template_plane=(GRAY_W, GRAY_H),
                 notes='Projective distortion varies across the field; GT is the projected quadrilateral.')

    @job('bl_gray_gloss_shadow')
    def _():
        s = Scene(768, 768, lights=('neutral', 'glossy_key')); m = s.texture_material(gray, roughness=.18, specular=.6)
        occluder = s.flat_material('Shadow caster', (.2, .24, .3))
        for i, pose in enumerate(grid_poses(2026102414, 3, 3, 256)):
            shadowed = i in (1, 4, 7)
            s.add_object(i, GRAY_W, GRAY_H, m, pose, identity='target', nominal_cast_shadow=shadowed, glossy=True)
            if shadowed:
                s.box(f'Caster_{i}', [pose['centre'][0] + 40, pose['centre'][1] - 30], (14, 120, 14), 60, occluder)
        s.render(output, 'bl_gray_gloss_shadow', template_plane=(GRAY_W, GRAY_H),
                 notes='Glossy targets under an oblique key light; boxes above objects 1, 4, 7 cast shadows.')

    @job('bl_sym_reference')
    def _():
        reference(output, 'bl_sym_reference', symmetric, GRAY_W, GRAY_H, (768, 768))

    @job('bl_sym_scene')
    def _():
        s = Scene(768, 768); m = s.texture_material(symmetric)
        for i, pose in enumerate(grid_poses(2026102415, 3, 3, 256)):
            s.add_object(i, GRAY_W, GRAY_H, m, pose, identity='target', rotational_symmetry_deg=180)
        s.render(output, 'bl_sym_scene', template_plane=(GRAY_W, GRAY_H),
                 notes='180-degree symmetric texture: angle and angle+180 are both correct.')

    records = {r['name']: r for r in meta['textures']}
    rgb_mix = ['red_base'] * 4 + ['red_hue+_8', 'red_hue-_8', 'red_chroma-_8', 'red_isogray0_8',
                                  'red_isogray90_4', 'red_hue+_4', 'red_lightness+_8', 'red_chroma-_4']
    rng = random.Random(rgb_order_seed); order = rgb_mix[:]; rng.shuffle(order)

    @job('bl_rgb_reference')
    def _():
        reference(output, 'bl_rgb_reference', textures / records['red_base']['file'], RGB_W, RGB_H, (768, 768))

    def rgb_scene(name, lights):
        s = Scene(1024, 768, lights=lights)
        materials = {n: s.texture_material(textures / records[n]['file']) for n in set(order)}
        for i, (pose, texture) in enumerate(zip(grid_poses(rgb_layout_seed, 4, 3, 256, scale_range=(.9, 1.1)), order)):
            r = records[texture]
            s.add_object(i, RGB_W, RGB_H, materials[texture], pose, identity='target', texture=texture,
                same_colour=r['same_colour'], direction=r['direction'], delta_e00_target=r['delta_e00_target'],
                delta_e00_achieved=r['delta_e00_achieved'], gray_texture_identical_to_base=r['gray_texture_identical_to_base'])
        s.render(output, name, template_plane=(RGB_W, RGB_H),
                 notes='Same layout and object colours in every bl_rgb scene; only the light changes.')

    @job('bl_rgb_white')
    def _():
        rgb_scene('bl_rgb_white', ('neutral',))

    @job('bl_rgb_warm3200')
    def _():
        rgb_scene('bl_rgb_warm3200', ('warm',))

    @job('bl_rgb_mixed')
    def _():
        rgb_scene('bl_rgb_mixed', ('left_warm', 'right_cool'))

    selected = only or list(jobs)
    for name in selected:
        if (output / f'{name}.png').exists():
            raise ValueError(f'refusing to overwrite {name}')
        jobs[name]()
    record = output / 'blender_generation.json'
    previous = json.loads(record.read_text())['rendered'] if record.exists() else []
    save(record, dict(script_sha256=sha(__file__), textures_json_sha256=sha(textures / 'textures.json'),
        blender_version=bpy.app.version_string, rendered=previous + selected, samples=SAMPLES, seed=SEED, threads=THREADS,
        rgb_order_seed=rgb_order_seed, rgb_layout_seed=rgb_layout_seed))


def main():
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--textures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', default='')
    parser.add_argument('--seed', type=int, default=SEED, help='Cycles sampling seed')
    parser.add_argument('--rgb-order-seed', type=int, default=RGB_ORDER_SEED, help='bl_rgb texture order')
    parser.add_argument('--rgb-layout-seed', type=int, default=RGB_LAYOUT_SEED, help='bl_rgb object poses')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    SEED = args.seed
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    build(args.textures.resolve(), out, [n for n in args.only.split(',') if n], args.rgb_order_seed, args.rgb_layout_seed)


if __name__ == '__main__':
    main()
