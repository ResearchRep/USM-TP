/*
    Copyright [2026] [Indian Institute of Science]
    SPDX-License-Identifier: Apache-2.0
*/

#include <string>
#include <map>
#include <chrono>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "ATen/ATen.h"
#include <c10/xpu/XPUStream.h>

using namespace dnnl;
typedef sycl::ext::oneapi::bfloat16 bf16;

// Convenience container to hold streams, devices, etc
struct {
    dnnl::engine cpu_engine, xpu_engine;
    dnnl::stream xpu_stream, cpu_stream;
    sycl::queue xpu_queue;

    struct {int sz; bf16 * ptr;} cpu_src = {.sz = 0}, cpu_dst = {.sz = 0};
} devices;

// Setup routine, must be called before doing anything else
extern "C"
void setup (sycl::queue * xpu_q) {
    // DNNL engine and stream for CPU
    devices.cpu_engine = dnnl::engine (engine::kind::cpu, 0);
    devices.cpu_stream = dnnl::stream (devices.cpu_engine);

    // DNNL engine and stream for XPU,
    // constructed from the passed SYCL queue
    devices.xpu_engine = sycl_interop::make_engine (
        xpu_q->get_device (), xpu_q->get_context ());
    devices.xpu_stream = sycl_interop::make_stream (
        devices.xpu_engine, *xpu_q);

    devices.xpu_queue = *xpu_q;
}

// Cleanup routine, should be called after doing everything
extern "C"
void setdown () {
    if (devices.cpu_src.sz > 0)
        sycl::free (devices.cpu_src.ptr, devices.xpu_queue);
    if (devices.cpu_dst.sz > 0)
        sycl::free (devices.cpu_dst.ptr, devices.xpu_queue);
}

// Launch a matrix multiplication using oneDNN
// src[M x K] x wt[K x N] -> dst[M x N]
auto launch_matmul (
    int M, int K, int N, int stride_N,
    void * src, void * wt, void * dst,
    int cpu
) {
    // Use the right engine and stream for this device.
    auto engn = cpu ?
        devices.cpu_engine : devices.xpu_engine;
    auto engine_stream = cpu ?
        devices.cpu_stream : devices.xpu_stream;

    // Dimensions
    memory::dims src_dims = {M, K};
    memory::dims wt_dims = {K, N};
    memory::dims dst_dims = {M, N};
    memory::dims dst_stride = {stride_N, 1};

    // Memory descriptors
    auto src_md = memory::desc(
        src_dims, memory::data_type::bf16, memory::format_tag::ab);
    auto wt_md = memory::desc(
        wt_dims, memory::data_type::bf16, memory::format_tag::ba);
    auto dst_md = memory::desc(
        dst_dims, memory::data_type::bf16, dst_stride);

    // Primitive
    auto matmul_pd = matmul::primitive_desc (
        engn, src_md, wt_md, dst_md);
    auto matmul_primitive = matmul (matmul_pd);

    // Memory objects
    auto src_mem = memory(src_md, engn, src);
    auto wt_mem = memory(wt_md, engn, wt);
    auto dst_mem = memory(dst_md, engn, dst);

    // Set primitive arguments
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, wt_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // Note that this call is blocking for the CPU
    // stream and asynchronous for the GPU stream
    return [=]() {
        matmul_primitive.execute (engine_stream, matmul_args);
    };
}

// Launch a matrix multiplication using ATen
void launch_matmul_at (
    int M, int K, int N, int stride_N,
    void * src, void * wt, void * dst,
    int cpu
) {
    auto dev = cpu ? "cpu" : "xpu";
    auto opts = at::TensorOptions ()
        .device (dev).dtype (c10::ScalarType::BFloat16);

    auto src_ten = at::from_blob (src, {M, K}, {K, 1}, opts);
    auto wt_ten = at::from_blob (wt, {N, K}, {K, 1}, opts);
    auto res_ten = at::from_blob (dst, {M, N}, {stride_N, 1}, opts);

    at::matmul_out (res_ten, src_ten, wt_ten.t ());
}

void enqueue_marker_kernel (bf16 * dst) {
    devices.xpu_queue.single_task ([=]() {
        *dst = 13;
    });
}

extern "C"
void enqueue_empty_kernel () {
    devices.xpu_queue.single_task ([]() {
        ;
    });
}

// Launch a matmul, split along the output's columns. Parameters:
// * M, K, N: matmul dimensions
// * src, wt, bias, dst: data buffers
// * cpu_N: number of output columns to be computed by CPU
// * flags: a string, where each character specifies a flag.
//   Possible flags:
//   - 'c': do not launch CPU partition
//   - 'g': do not launch GPU partition
//   - 'e': empty CPU partition
//          (i.e. host task with empty function)
//   - 'u': do not assume USM shared allocations.
//          When specified, extra data copy ops are used,
//          and CPU partition reads weights from `wt_cpu`.
//   - 'm': enqueue a marker kernel after the GPU GEMV
//
extern "C"
void split_matmul (
    int M, int K, int N,
    bf16 * src, bf16 * wt, bf16 * bias, bf16 * dst,
    int cpu_N, char * flags, bf16 * wt_cpu
) {
    bool no_cpu = false, no_gpu = false, empty_cpu = false, no_usm = false,
         marker = false, no_aten = false;
    for (char * c = flags; *c != '\0'; c++) {
        switch (*c) {
            case 'c': no_cpu = true; break;
            case 'g': no_gpu = true; break;
            case 'e': empty_cpu = true; break;
            case 'u': no_usm = true; break;
            case 'm': marker = true; break;
            case 'a': no_aten = true; break;
            default:
                printf ("WARNING: split_matmul: unknown flag '%c'\n", *c);
                break;
        }
    }

    // Partition weights
    bf16 * wt_a = wt;
    bf16 * dst_a = dst;
    bf16 * wt_b = wt + K * (cpu_N);    // skip cpu_N columns of length K
    bf16 * dst_b = dst + cpu_N;

    {
        // Will be passed to CPU partition
        int src_sz = M * K * sizeof (bf16);
        int dst_a_sz = M * cpu_N * sizeof (bf16);
        struct {bf16 * src; bf16 * wt; bf16 * dst; } cpu_bufs;
        if (! no_usm) {
            cpu_bufs.src = src;
            cpu_bufs.dst = dst_a;
            cpu_bufs.wt = wt_a;
        } else {
            if (wt_cpu == NULL) {
                printf ("WARNING: flag 'u' passed, but wt_cpu is NULL\n");
            }
            if (M != 1) {
                printf ("WARNING: flag 'u' does not work with M != 1; "
                        "Skipping matmul\n");
                return;
            }
            // Check if our cached buffers are large enough
            // If not, free and allocate again
            if (devices.cpu_src.sz < src_sz) {
                sycl::free (devices.cpu_src.ptr, devices.xpu_queue);
                devices.cpu_src.ptr = (bf16 *) sycl::malloc_host (src_sz, devices.xpu_queue);
                devices.cpu_src.sz = src_sz;
            }
            if (devices.cpu_dst.sz < dst_a_sz) {
                sycl::free (devices.cpu_dst.ptr, devices.xpu_queue);
                devices.cpu_dst.ptr = (bf16 *) sycl::malloc_host (dst_a_sz, devices.xpu_queue);
                devices.cpu_dst.sz = dst_a_sz;
            }
            // Copy `src` to CPU memory
            devices.xpu_queue.memcpy (devices.cpu_src.ptr, src, src_sz);
            cpu_bufs.wt = wt_cpu;
            cpu_bufs.src = devices.cpu_src.ptr;
            cpu_bufs.dst = devices.cpu_dst.ptr;
        }
        auto cpu_work = launch_matmul (M, K, cpu_N, N, cpu_bufs.src, cpu_bufs.wt, cpu_bufs.dst, 1);

        devices.xpu_queue.wait ();
        // Compute starts here
        if (! no_gpu) {
            if (no_aten)
                launch_matmul (M, K, N - cpu_N, N, src, wt_b, dst_b, 0)();
            else
                launch_matmul_at (M, K, N - cpu_N, N, src, wt_b, dst_b, 0);
            if (marker) {
                enqueue_marker_kernel (dst);
            }
        }
        if (! no_cpu) {
            if (! empty_cpu)
                cpu_work ();
        }

        if (no_usm) {
            // Copy `dst_a` to GPU memory
            devices.xpu_queue.memcpy (dst_a, devices.cpu_dst.ptr, dst_a_sz);
        }
    }
}

// Qwen3 MLP forward pass, split over GPU and CPU.
// The MLP is parametrized by the hidden and intermediate dimensions
// (H and I respectively). Conceptually, the computation
// (except for an element-wise addition at the end) can be viewed
// as I independent computations. The CPU handles cpuI of these
// computations, while the GPU handles the the remaining (I - cpuI).
//
// Parameters:
// * H, I, cpuI: see above
// * src: input activations
// * gate_wt, up_wt, down_wt: weights for gate-, up- and down-projection
// * dst: output buffer
// As with `split_matmul`, some flags are accepted here:
// - 'c': don't launch CPU partition
// - 'g': don't launch GPU partition
//
extern "C"
void split_qwen3_mlp_fwd (
    int H, int I, int cpuI,
    uint8_t * src, uint8_t * gate_wt, uint8_t * up_wt, uint8_t * down_wt,
    uint8_t * dst,
    char * flags
) {
    // According to HF transformers:
    // down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #define bf16_ab memory::data_type::bf16, memory::format_tag::ab
    #define bf16_ba memory::data_type::bf16, memory::format_tag::ba

    bool
        no_cpu = false,
        no_gpu = false;
    for (char * c = flags; *c != '\0'; c++) {
        switch (*c) {
            case 'c': no_cpu = true; break;
            case 'g': no_gpu = true; break;
            default:
                printf ("WARNING: split_matmul: unknown flag '%c'\n", *c);
                break;
        }
    }

    // CPU and GPU engines
    auto geng = devices.xpu_engine;
    auto ceng = devices.cpu_engine;
    auto gstr = devices.xpu_stream;
    auto cstr = devices.cpu_stream;

    // Pointer arithmetic to split the weights
    auto
        gateA_wt =  gate_wt,
        gateB_wt =  gate_wt + (cpuI*H)*sizeof(bf16),
        upA_wt =    up_wt,
        upB_wt =    up_wt + (cpuI*H)*sizeof(bf16),
        downA_wt =  down_wt,
        downB_wt =  down_wt + cpuI*sizeof(bf16);

    // Memory descriptors
    memory::desc
        src_md  ({1, H}, bf16_ab),
        down_md ({I, H}, bf16_ba),
        dst_md  ({1, H}, bf16_ab),
        t1_md   ({1, I}, bf16_ab),
        t2_md   ({1, I}, bf16_ab);
    // Split memory descriptors
    memory::desc
        gateA_md    ({H, cpuI}, bf16_ba),
        gateB_md    ({H, I-cpuI}, bf16_ba),
        upA_md      ({H, cpuI}, bf16_ba),
        upB_md      ({H, I-cpuI}, bf16_ba),
        t1A_md      ({1, cpuI}, bf16_ab),
        t1B_md      ({1, I-cpuI}, bf16_ab),
        t2A_md      ({1, cpuI}, bf16_ab),
        t2B_md      ({1, I-cpuI}, bf16_ab),
        downA_md    ({cpuI, H}, memory::data_type::bf16, {1, I}),
        downB_md    ({I-cpuI, H}, memory::data_type::bf16, {1, I});

    // Memory objects
    memory
        src_mem     (src_md, geng, src),
        srcA_mem    (src_md, ceng, src),
        down_mem    (down_md, geng, down_wt),
        downA_mem   (downA_md, ceng, downA_wt),
        downB_mem   (downB_md, geng, downB_wt),
        dst_mem     (dst_md, geng, dst),
        dstA_mem    (dst_md, geng),
        t1_mem      (t1_md, geng),
        t2_mem      (t2_md, geng);

    // Even more pointer arithmetic
    uint8_t
        * t1_ptr = (uint8_t *) t1_mem.get_data_handle (),
        * t2_ptr = (uint8_t *) t2_mem.get_data_handle ();
    auto
        t1A_ptr = t1_ptr,
        t1B_ptr = t1_ptr + cpuI*sizeof(bf16),
        t2A_ptr = t2_ptr,
        t2B_ptr = t2_ptr + cpuI*sizeof(bf16);

    // Split memory objects
    memory
        gateA_mem   (gateA_md, ceng, gateA_wt),
        gateB_mem   (gateB_md, geng, gateB_wt),
        upA_mem     (upA_md, ceng, upA_wt),
        upB_mem     (upB_md, geng, upB_wt),
        t1A_mem     (t1A_md, ceng, t1A_ptr),
        t1B_mem     (t1B_md, geng, t1B_ptr),
        t2A_mem     (t2A_md, ceng, t2A_ptr),
        t2B_mem     (t2B_md, geng, t2B_ptr);

    // Primitive descriptors and attributes
    // Post-ops for gate projection; covers SiLU
    dnnl::post_ops gpo;
    gpo.append_eltwise (algorithm::eltwise_swish, 1.0f, 0.0f);
    dnnl::primitive_attr ga;
    ga.set_post_ops (gpo);
    // Post-ops for up projection; covers element-wise mulplication
    dnnl::post_ops upo;
    upo.append_binary (algorithm::binary_mul, t1_md);
    dnnl::primitive_attr ua;
    ua.set_post_ops (upo);

    // Primitive descriptors
    // Note that the element-wise mul is fused with up-projection
    // only for the GPU (partition B).
    // Doing so for CPU corrupts output for an unidentified reason,
    // but not doing so doesn't harm performance much.
    matmul::primitive_desc
        gateA_pd    (ceng, src_md, gateA_md, t1A_md, ga),
        gateB_pd    (geng, src_md, gateB_md, t1B_md, ga),
        upA_pd      (ceng, src_md, upA_md, t1A_md/*, ua*/),
        upB_pd      (geng, src_md, upB_md, t1B_md, ua),
        down_pd     (geng, t2_md, down_md, dst_md),
        downA_pd    (ceng, t2A_md, downA_md, dst_md),
        downB_pd    (geng, t2B_md, downB_md, dst_md);
    binary::primitive_desc
        mulA_pd (ceng, algorithm::binary_mul, t2A_md, t1A_md, t2A_md),
        add_pd  (geng, algorithm::binary_add, dst_md, dst_md, dst_md);

    // Primitives
    matmul
        gateA_prim  {gateA_pd},
        gateB_prim  {gateB_pd},
        upA_prim    {upA_pd},
        upB_prim    {upB_pd},
        downA_prim  {downA_pd},
        downB_prim  {downB_pd},
        down_prim   {down_pd};
    binary
        mulA_prim   (mulA_pd),
        add_prim    (add_pd);

    // Run!
   if (! no_cpu)
    devices.xpu_queue.wait ();

    // Partition B
   if (! no_gpu) {
    gateB_prim.execute (gstr, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, gateB_mem},
        {DNNL_ARG_DST, t1B_mem},
    });
    upB_prim.execute (gstr, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, upB_mem},
        {DNNL_ARG_DST, t2B_mem},
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, t1B_mem},
    });
    downB_prim.execute (gstr, {
        {DNNL_ARG_SRC, t2B_mem},
        {DNNL_ARG_WEIGHTS, downB_mem},
        {DNNL_ARG_DST, dst_mem},
    });
   }
   if (! no_cpu) {
    // Partition A
    gateA_prim.execute (cstr, {
        {DNNL_ARG_SRC, srcA_mem},
        {DNNL_ARG_WEIGHTS, gateA_mem},
        {DNNL_ARG_DST, t1A_mem},
    });
    upA_prim.execute (cstr, {
        {DNNL_ARG_SRC, srcA_mem},
        {DNNL_ARG_WEIGHTS, upA_mem},
        {DNNL_ARG_DST, t2A_mem},
    });
    mulA_prim.execute (cstr, {
        {DNNL_ARG_SRC_0, t2A_mem},
        {DNNL_ARG_SRC_1, t1A_mem},
        {DNNL_ARG_DST, t2A_mem},
    });
    downA_prim.execute (cstr, {
        {DNNL_ARG_SRC, t2A_mem},
        {DNNL_ARG_WEIGHTS, downA_mem},
        {DNNL_ARG_DST, dstA_mem},
    });
   }
    // Common
    add_prim.execute (gstr, {
        {DNNL_ARG_SRC_0, dst_mem},
        {DNNL_ARG_SRC_1, dstA_mem},
        {DNNL_ARG_DST, dst_mem},
    });
}

extern "C"
unsigned long benchmark_gemv (
    int K, int N,
    void * src, void * wt, void * dst,
    char * variant, int reps
) {
    std::function<void(void)> work, wait;

    if      (! strcmp (variant, "dnnl_gpu")) {
        printf ("OneDNN, GPU\n");
        work = launch_matmul (1, K, N, N, src, wt, dst, 0);
        wait = [=]() { devices.xpu_queue.wait ();};
    }
    else if (! strcmp (variant, "dnnl_cpu")) {
        printf ("OneDNN, CPU\n");
        work = launch_matmul (1, K, N, N, src, wt, dst, 1);
        wait = [=]() {};
    }
    else if (! strcmp (variant, "aten_gpu")) {
        printf ("ATen, GPU\n");
        work = [=]() { launch_matmul_at (1, K, N, N, src, wt, dst, 0); };
        wait = [=]() { devices.xpu_queue.wait (); };
    }
    else
        printf ("WARNING: Invalid variant: %s\n", variant);

    printf ("%d reps\n", reps);

    struct timespec start, end;
    wait ();
    clock_gettime (CLOCK_MONOTONIC, &start);
    for (int i = 0; i < reps; i++)
        work ();
    wait ();
    clock_gettime (CLOCK_MONOTONIC, &end);

    unsigned long elapsed_ns =
        (end.tv_sec - start.tv_sec) * 1000000000 +
        (end.tv_nsec - start.tv_nsec);

    return elapsed_ns;
}
