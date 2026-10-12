#!/bin/sh -e

variant="$1"
case "$variant" in
    tbb)        rt=TBB;;
    threadpool) rt=THREADPOOL;;
    omp)        rt=OMP;;
    *)   echo "Invalid variant"; false;;
esac

[ -e oneDNN ] || git clone https://github.com/uxlfoundation/oneDNN --depth 1
cd oneDNN

buildd="autobuild-$variant"
mkdir -p $buildd; cd $buildd
rm -f CMakeCache.txt

. /opt/intel/oneapi/2025.3/oneapi-vars.sh
export CC=icx
export CXX=icpx

cmake \
    -DONEDNN_GPU_RUNTIME=SYCL \
    -DONEDNN_CPU_RUNTIME=$rt \
    -DONEDNN_BUILD_TESTS=OFF \
    -DONEDNN_BUILD_EXAMPLES=OFF \
    -DONEDNN_LIBRARY_NAME="dnnl-$variant" \
    ..

cmake --build . --parallel 8
cmake --install . --prefix ../install
