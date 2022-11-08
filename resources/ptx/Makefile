mts_include_folder = ../../include
mts_shape_folder   = ../../src/shapes/optix
mts_optix_main     = ../../src/render/optix/optix_rt.cu

OPTIX_PATH := $(or $(OPTIX_PATH),/home/nroussel/rgl/nvidia-optix/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/include/)
#OPTIX_PATH := $(or $(OPTIX_PATH),/opt/optix/include/)

all: optix_rt.ptx

optix_rt.ptx: $(mts_optix_main) $(mts_include_folder)/* $(mts_shape_folder)/*
	/usr/local/cuda-11.7/bin/nvcc $(mts_optix_main) \
		-G -O0 \
		 -I $(mts_include_folder) -I $(mts_shape_folder) -I $(OPTIX_PATH) \
		 -gencode arch=compute_61,code=compute_61 --ptx

clean:
	rm -f optix_rt.ptx
