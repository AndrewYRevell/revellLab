#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:25:40 2021

@author: arevell
"""

# Blender Python script for converting a mesh to GLB with Draco compression.
# Tested on Blender 2.82
# Usage:
#   blender --background --factory-startup --addons io_scene_gltf2 --python blender_compress_mesh.py -- -i #{source_path} -o #{out_path}

from os import path
from contextlib import redirect_stdout
from sys import argv
import argparse
import io
import bpy
import bpy_types

def file_name(filepath):
    return path.split(filepath)[1]

def dir_path(filepath):
    return path.split(filepath)[0]

def file_suffix(filepath):
    return path.splitext(file_name(filepath))[1]

def import_func_wrapper(func, filepath):
    func(filepath=filepath)

def import_mesh(filepath):
    import_func = {
        '.obj': bpy.ops.import_scene.obj,
        '.ply': bpy.ops.import_mesh.ply,
        '.stl': bpy.ops.import_mesh.stl,
        '.wrl': bpy.ops.import_scene.x3d,
        '.x3d': bpy.ops.import_scene.x3d,
        '.glb': bpy.ops.import_scene.gltf,
        '.gltf': bpy.ops.import_scene.gltf
    }

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        import_func_wrapper(import_func[file_suffix(filepath)], filepath=filepath)
        stdout.seek(0)
        return stdout.read()

if "--" not in argv:
    argv = [] # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]
parser = argparse.ArgumentParser(description='Blender mesh file to GLB conversion tool')
parser.add_argument('-i', '--input', help='mesh file to be converted')
parser.add_argument('-o', '--output', help='output GLB file')
args = parser.parse_args(argv)

if (args.input and args.output):
    ifile = args.input
    ofile = args.output

    err_msg = ''
    try: 
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        if len(bpy.data.objects) == 0:
            stdout = import_mesh(ifile)
            if len(bpy.data.objects) != 0:
                for obj in bpy.data.objects:
                    if type(obj.data) == bpy_types.Mesh:
                        # Apply material (only works in Blender 2.8)
                        if not obj.data.materials:
                             mat = bpy.data.materials.new(name="Material")
                             mat.use_nodes = True
                             obj.data.materials.append(mat)

                        bpy.context.view_layer.objects.active = obj
                        
                bpy.ops.object.convert(target='MESH')

                #bpy.ops.object.origin_set()
                bpy.ops.export_scene.gltf(filepath=ofile, export_yup = False)
            else:
                # likely invalid file error, not an easy way to capture this from Blender
                err_msg = stdout.replace("\n", "; ")
        else:
            err_msg = 'Error deleting Blender scene objects'
    except Exception as e:
        err_msg = str(e).replace("\n", "; ")
else:
    err_msg = 'Command line arguments not supplied or inappropriate'

if err_msg:
    raise ValueError(err_msg)
else:
    print('Successfully converted')