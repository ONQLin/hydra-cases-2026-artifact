// Event-driven, uniform-link network with cut-through packet aggregation.
#pragma once

#include <cstdint>
#include <deque>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

namespace hydra {
using Tick = uint64_t;
struct Completion { uint64_t flow, tick; };
struct Statistics {
    uint64_t events = 0, packets = 0, submitted = 0, completed = 0;
    uint64_t received_bytes = 0, peak_buffer_bytes = 0, blocked_checks = 0;
};

class PacketNetwork {
  public:
    PacketNetwork(int rows, int columns, uint64_t flit_bytes, Tick period,
                  Tick hop_delay, uint64_t quantum, uint64_t buffer_bytes,
                  const double* hbm_bandwidth, bool dma, uint64_t dma_burst,
                  Tick access_delay);
    void submit(uint64_t id, int source, int destination, uint64_t bytes);
    std::vector<Completion> advance(Tick target);
    Tick now() const { return now_; }
    size_t active() const { return flows_.size(); }
    const Statistics& statistics() const { return stats_; }

  private:
    struct Packet { uint64_t flow, bytes; size_t hop; };
    struct Port {
        std::deque<Packet> queue;
        uint64_t occupied = 0;
        bool busy = false;
        int last_grant = -1;
    };
    struct Source {
        std::deque<uint64_t> ready, dma_queue;
        double bandwidth;
        bool dma_busy = false;
    };
    struct Flow {
        int source;
        std::vector<int> path;
        uint64_t outstanding, ready_bytes = 0, dma_remaining;
        Tick ready_at;
        bool queued = false;
    };
    enum class Kind { Arrive, Depart, Deliver, DmaDone };
    struct Event {
        Tick tick;
        uint64_t sequence;
        Kind kind;
        int port;
        Packet packet;
        bool operator<(const Event& other) const {
            return tick != other.tick ? tick > other.tick : sequence > other.sequence;
        }
    };
    int rows_, columns_, nodes_;
    uint64_t flit_bytes_, quantum_, buffer_bytes_, dma_burst_;
    Tick period_, hop_delay_, access_delay_, now_ = 0;
    bool dma_;
    uint64_t sequence_ = 0;
    Statistics stats_;
    std::vector<Port> ports_;
    std::vector<Source> sources_;
    std::vector<std::vector<int>> contenders_;
    std::set<int> active_ports_, active_sources_;
    std::vector<int> candidates_, targets_;
    std::vector<std::vector<int>> links_;
    std::unordered_map<uint64_t, Flow> flows_;
    std::priority_queue<Event> events_;
    std::vector<Completion> completions_;

    std::vector<int> route(int source, int destination) const;
    void schedule(Tick delay, Kind kind, int port, Packet packet);
    void reserve(int port, uint64_t bytes);
    void ready(uint64_t id, uint64_t bytes);
    void pump();
    void process(const Event& event);
    void start_packet(int port);
};
} // namespace hydra
