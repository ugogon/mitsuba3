import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

mi.set_log_level(mi.LogLevel.Info)
dr.set_log_level(dr.LogLevel.Info)

dr.set_flag(dr.JitFlag.LaunchBlocking, True)
dr.set_flag(dr.JitFlag.VCallRecord, True)
dr.set_flag(dr.JitFlag.LoopRecord, False)

dr.set_flag(dr.JitFlag.KernelHistory, True)
dr.set_flag(dr.JitFlag.PrintIR, True)

scene_description = {
    'type': 'scene',
    'sensor' : {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, -20], target=[0, 0, 0], up=[0, 1, 0])
    },
    'cube': {
        'type': 'cube'
    },
    'emitter': {
        'type': 'constant'
    },
    'integrator': {
        'type': 'direct',
        'hide_emitters': True,
        'emitter_samples': 0,
        'bsdf_samples': 1,
    }
}

scene = mi.load_dict(scene_description)

dr.kernel_history()
tensor = mi.render(scene, spp=1)

print("DONE RENDERING")
print("DONE RENDERING")
print("DONE RENDERING")

out = dr.kernel_history()
print(f"{out=}")

mi.Bitmap(tensor)
