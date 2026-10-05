// Stable C ABI: exceptions never cross the Python/C++ boundary.
#include "packet_network.hh"
#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
template<class Action> int guarded(Action action) {
    try { action(); last_error.clear(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "Unknown C++ network error"; return -1; }
}
}
extern "C" {
int hp_abi_version() { return 1; }
const char* hp_error() { return last_error.c_str(); }
void* hp_create(int rows, int columns, uint64_t flit_bytes, uint64_t period,
        uint64_t hop_delay, uint64_t quantum, uint64_t buffer_bytes,
        const double* bandwidth, int dma, uint64_t dma_burst, uint64_t access) {
    std::unique_ptr<hydra::PacketNetwork> network;
    const int result = guarded([&] {
        if (rows <= 0 || columns <= 0 || rows > 4096 || columns > 4096 ||
            uint64_t(rows) * columns > 4096 || !bandwidth)
            throw std::invalid_argument("Invalid mesh dimensions or HBM array");
        network = std::make_unique<hydra::PacketNetwork>(rows, columns, flit_bytes,
            period, hop_delay, quantum, buffer_bytes, bandwidth, dma != 0, dma_burst, access);
    });
    return result ? nullptr : network.release();
}
void hp_destroy(void* handle) { delete static_cast<hydra::PacketNetwork*>(handle); }
int hp_submit(void* handle, uint64_t id, int source, int destination, uint64_t bytes) {
    return guarded([&] {
        if (!handle) throw std::invalid_argument("Network is closed");
        static_cast<hydra::PacketNetwork*>(handle)->submit(id, source, destination, bytes);
    });
}
int hp_advance(void* handle, uint64_t target, hydra::Completion* output,
        uint64_t capacity, uint64_t* count, uint64_t* reached) {
    return guarded([&] {
        if (!handle || !count || !reached) throw std::invalid_argument("Invalid advance arguments");
        auto& network = *static_cast<hydra::PacketNetwork*>(handle);
        if (capacity < network.active() || (capacity && !output))
            throw std::invalid_argument("Completion array is too small");
        const auto completed = network.advance(target);
        std::copy(completed.begin(), completed.end(), output);
        *count = completed.size();
        *reached = network.now();
    });
}
int hp_stats(void* handle, hydra::Statistics* output) {
    return guarded([&] {
        if (!handle || !output) throw std::invalid_argument("Invalid statistics arguments");
        *output = static_cast<hydra::PacketNetwork*>(handle)->statistics();
    });
}
}
