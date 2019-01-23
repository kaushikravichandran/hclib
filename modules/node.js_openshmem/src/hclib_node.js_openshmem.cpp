
#include <shmem.h>
#include "hclib_node.js_openshmem.h"
#include "hclib-locality-graph.h"

int nic_locale_id = -1;
hclib::locale_t *nic = NULL;

HCLIB_MODULE_INITIALIZATION_FUNC(nodejs_openshmem_pre_initialize) {
    nic_locale_id = hclib_add_known_locale_type("Interconnect");
    HASSERT(nic_locale_id > -1);
}

int shmem_init_thread(int, int*);

HCLIB_MODULE_INITIALIZATION_FUNC(nodejs_openshmem_post_initialize) {

#ifdef USE_CRAY_SHMEM_7
    int ret = ::shmem_init_thread(SHMEM_THREAD_MULTIPLE);
    assert(ret == SHMEM_THREAD_MULTIPLE);
#else
    int major, minor;
    shmem_info_get_version(&major, &minor);
    if(major>=1 && minor>=4) {
        int provided;
        int ret = ::shmem_init_thread(SHMEM_THREAD_MULTIPLE, &provided);
        assert(ret == 0);
        assert(provided == SHMEM_THREAD_MULTIPLE);
    }
    else {
        printf("WARNING: SHMEM 1.4 or above required\n");
        ::shmem_init();
    }
#endif

    int n_nics;
    hclib::locale_t **nics = hclib::get_all_locales_of_type(nic_locale_id,
            &n_nics);
    HASSERT(n_nics == 1);
    HASSERT(nics);
    HASSERT(nic == NULL);
    nic = nics[0];
}

HCLIB_MODULE_INITIALIZATION_FUNC(nodejs_openshmem_finalize) {
    ::shmem_finalize();
}

int hclib::shmem_my_pe() {
    return ::shmem_my_pe();
}

int hclib::shmem_n_pes() {
    return ::shmem_n_pes();
}

HCLIB_REGISTER_MODULE("node.js_openshmem", nodejs_openshmem_pre_initialize, nodejs_openshmem_post_initialize, nodejs_openshmem_finalize)