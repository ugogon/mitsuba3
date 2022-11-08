import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_rgb')

mi.set_log_level(mi.LogLevel.Trace)
dr.set_log_level(dr.LogLevel.Trace)

dr.set_flag(dr.JitFlag.VCallRecord, True)
dr.set_flag(dr.JitFlag.KernelHistory, True)
dr.set_flag(dr.JitFlag.PrintIR, True)

scene_description = {
    'type': 'scene',
    'cube': {
        'type': 'cube'
    },
    'emitter': {
        'type': 'constant'
    },
}

scene = mi.load_dict(scene_description)

ray_o = mi.Point3f([0, 0, -20])
ray_d = mi.Vector3f([0, 0, 20])
ray = mi.Ray3f(o=ray_o, d=ray_d)

#breakpoint()
si = scene.ray_intersect(ray)
dr.eval(si)
