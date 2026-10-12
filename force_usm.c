/*
    Copyright [2026] [Indian Institute of Science]
    SPDX-License-Identifier: Apache-2.0
*/

#include <level_zero/ze_api.h>
#include "level_zero/layers/zel_tracing_api.h"
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define ZE_CALL(expr) { \
    ze_result_t status = ZE_RESULT_SUCCESS; \
    status = expr; \
    assert((#expr, status == ZE_RESULT_SUCCESS)); \
}

void on_exit_zemgap (
    ze_mem_get_alloc_properties_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
) {
    // Spoof properties to make shared alloc look like device alloc
    ze_memory_allocation_properties_t * props = *params->ppMemAllocProperties;
    if (props->type == ZE_MEMORY_TYPE_SHARED)
        props->type = ZE_MEMORY_TYPE_DEVICE;
}

struct zemad_alloc_info {
    void * toy_ptr;   // What will eventually go to the user
};
void on_enter_zemad (
    ze_mem_alloc_device_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
) {
    struct zemad_alloc_info * instance_data =
        malloc (sizeof (struct zemad_alloc_info));
    *ppTracerInstanceUserData = instance_data;

    // Allocate shared memory of the requested size
    ze_host_mem_alloc_desc_t host_desc = {};
    ZE_CALL(zeMemAllocShared (
        *(params->phContext),
        *(params->pdevice_desc),
        &host_desc,
        *(params->psize),
        *(params->palignment),
        *(params->phDevice),
        *(params->ppptr)
    ));

    // Modify user's call to request 1 byte
    // (since 0 bytes cannot be requested)
    *(params->psize) = 1;
    *(params->ppptr) = &(instance_data->toy_ptr);
}
void on_exit_zemad (
    ze_mem_alloc_device_params_t* params,
    ze_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
) {
    assert (result == ZE_RESULT_SUCCESS);
    struct zemad_alloc_info * instance_data =
        * ppTracerInstanceUserData;

    // Free the 1-byte pointer
    ZE_CALL(zeMemFree (
        *(params->phContext),
        instance_data->toy_ptr
    ));
    free (instance_data);
}

ze_driver_handle_t drivers [1];
zel_tracer_handle_t tracer;
void setup () {
    // Level zero driver and context init
    uint32_t count = 0;
    ze_init_driver_type_desc_t desc = {};
    desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
    ZE_CALL(zeInitDrivers (&count, NULL, &desc));
    assert(("Expected 1 driver", count == 1));
    ZE_CALL(zeInitDrivers (&count, drivers, &desc));

    // Tracer handle setup
    zel_tracer_desc_t tracer_desc = {
        ZEL_STRUCTURE_TYPE_TRACER_DESC, NULL,
        NULL /* global user data */
    };
    tracer = NULL;
    ZE_CALL(zelTracerCreate(&tracer_desc, &tracer));

    // Callback registration
    zel_core_callbacks_t epilogue_callbacks = {};
    epilogue_callbacks.Mem.pfnGetAllocPropertiesCb = on_exit_zemgap;
    epilogue_callbacks.Mem.pfnAllocDeviceCb = on_exit_zemad;
    ZE_CALL(zelTracerSetEpilogues(tracer, &epilogue_callbacks));

    zel_core_callbacks_t prologue_callbacks = {};
    prologue_callbacks.Mem.pfnAllocDeviceCb = on_enter_zemad;
    ZE_CALL(zelTracerSetPrologues(tracer, &prologue_callbacks));

    // Enable tracing
    ZE_CALL(zelTracerSetEnabled(tracer, true));
}
void cleanup () {
    ZE_CALL(zelTracerSetEnabled(tracer, false));
    ZE_CALL(zelTracerDestroy(tracer));
}
