#define USE_RESILIENT_PROMISE

#include "hclib_cpp.h"
#include "hclib_resilience.h"
#include "hclib_resilience_diamond_mpi.h"
#include <unistd.h>

namespace diamond = hclib::resilience::diamond;

enum TASK_STATE {NON_LEAF, LEAF};

//int_obj is not required for replay promises, base types can be used.
//Here it is just used to print inside constructor/destructor
class int_obj : public checkpoint::obj {
  public:
    int n;
    int_obj() { printf("creating int_obj\n"); }
    ~int_obj() { printf("deleting int_obj\n"); }

    bool equals(hclib::resilience::obj* obj2) {
      return n == ((int_obj*)obj2)->n;
    }

    void deserialize(checkpoint::archive_obj* ar_ptr) {
        n = *(int*)(ar_ptr->data);
    }

    checkpoint::archive_obj* serialize() {
        auto ar_ptr = new checkpoint::archive_obj();
        ar_ptr->size = sizeof(int);
        ar_ptr->data = malloc(ar_ptr->size);
        memcpy(ar_ptr->data, &n, ar_ptr->size);
        return ar_ptr;
    }
};

int main(int argc, char ** argv) {
    int SIGNAL_VALUE = 42;
    const char *deps[] = { "system"};
    //MPI_Init(&argc, &argv);
    //int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    //assert(provided == MPI_THREAD_MULTIPLE);
    hclib::launch(deps, 1, [=]() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        hclib::finish([=]() {
             auto prom = new hclib::promise_t<int*>();
             auto prom1 = new diamond::promise_t<int_obj*>(1);
             auto prom_res = new hclib::promise_t<int>();
             auto prom_recv = new hclib::promise_t<int_obj*>();
             auto prom_send = new hclib::promise_t<int_obj*>();

            hclib::async_await( [=]() {
                   sleep(1);
                   int *n= (int*) malloc(sizeof(int));
                   *n = SIGNAL_VALUE;
                   prom->put(n);
            }, future_nullptr);

            hclib::async_await( [=]() {
                    int_obj *n2_tmp = prom1->get_future()->get();
                    printf("Value2 %d\n", n2_tmp->n);
                    //prom1->get_future()->release();
            }, prom1->get_future());

            if(rank == 0)
                hclib::async_await( [=]() {
                        int_obj *send_tmp =  prom_send->get_future()->get();
                        printf("Value send %d\n", send_tmp->n);
                }, prom_send->get_future());
            else if(rank == 1)
                hclib::async_await( [=]() {
                      int_obj *recv_tmp = prom_recv->get_future()->get();
                      printf("Value Recv %d\n", recv_tmp->n);
                }, prom_recv->get_future());

            //This could be an array, vector, hash table or anything
            diamond::async_await_check<NON_LEAF>( [=]() {
                    int* signal = prom->get_future()->get();
                    assert(*signal == SIGNAL_VALUE);
                    printf("Value1 %d diamond %d\n", *signal, diamond::get_index());

                    diamond::async_await( [=]() {
                        int_obj *n2 = new int_obj();
                        n2->n = 22;
                        prom1->put(n2);
                        if(rank == 0) {
                            diamond::Isend(n2, 1, 1, 0, prom_send);
                        }
                    }, future_nullptr);

                    if(rank == 1) {
                        int buf=6;
                        diamond::Irecv(sizeof(int), 0, 1, prom_recv);
                        printf("Received %d in rank %d\n", buf, rank);
                    }
            }, prom_res, prom->get_future());

            hclib::async_await( [=]() {
                    int res = prom_res->get_future()->get();
                    printf("result %d\n", res);
                    if(res == 0) exit(0);
            }, prom_res->get_future());
        });
    });
    printf("Going to Exiting\n");
    //MPI_Finalize();
    printf("Exiting...\n");
    return 0;
}
