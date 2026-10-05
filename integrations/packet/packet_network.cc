#include "packet_network.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hydra {
PacketNetwork::PacketNetwork(int rows, int columns, uint64_t flit_bytes,
        Tick period, Tick hop_delay, uint64_t quantum, uint64_t buffer_bytes,
        const double* bandwidth, bool dma, uint64_t dma_burst, Tick access_delay)
    : rows_(rows), columns_(columns), nodes_(rows * columns),
      flit_bytes_(flit_bytes), quantum_(quantum), buffer_bytes_(buffer_bytes),
      dma_burst_(dma_burst), period_(period), hop_delay_(hop_delay),
      access_delay_(access_delay), dma_(dma) {
    if (rows <= 0 || columns <= 0 || nodes_ > 4096 || !flit_bytes || !period ||
        !hop_delay || quantum < flit_bytes || buffer_bytes < quantum ||
        quantum % flit_bytes || (dma && (dma_burst < flit_bytes || dma_burst % flit_bytes)))
        throw std::invalid_argument("Invalid mesh, timing, packet quantum, or buffer configuration");
    // Each endpoint has separate injection and ejection serialization resources.
    ports_.resize(2 * nodes_);
    links_.assign(nodes_, std::vector<int>(nodes_, -1));
    for (int node = 0; node < nodes_; ++node) {
        if (!std::isfinite(bandwidth[node]) || bandwidth[node] < 0)
            throw std::invalid_argument("HBM bandwidth must be finite and nonnegative");
        sources_.push_back({{}, {}, bandwidth[node], false});
        for (int other = 0; other < nodes_; ++other) {
            if (std::abs(node / columns_ - other / columns_) +
                std::abs(node % columns_ - other % columns_) == 1) {
                links_[node][other] = ports_.size();
                ports_.emplace_back();
            }
        }
    }
}

std::vector<int> PacketNetwork::route(int source, int destination) const {
    std::vector<int> path{source};
    int current = source;
    // Deterministic XY routing on row-major node IDs.
    while (current % columns_ != destination % columns_) {
        const int next = current + (current % columns_ < destination % columns_ ? 1 : -1);
        path.push_back(links_[current][next]);
        current = next;
    }
    while (current / columns_ != destination / columns_) {
        const int next = current + (current < destination ? columns_ : -columns_);
        path.push_back(links_[current][next]);
        current = next;
    }
    path.push_back(nodes_ + destination);
    return path;
}

void PacketNetwork::schedule(Tick delay, Kind kind, int port, Packet packet) {
    if (delay > std::numeric_limits<Tick>::max() - now_)
        throw std::overflow_error("Network tick overflow");
    events_.push({now_ + delay, sequence_++, kind, port, packet});
}

void PacketNetwork::reserve(int port, uint64_t bytes) {
    auto& buffer = ports_.at(port);
    if (bytes > buffer_bytes_ - buffer.occupied)
        throw std::logic_error("Network buffer overflow");
    buffer.occupied += bytes;
    stats_.peak_buffer_bytes = std::max(stats_.peak_buffer_bytes, buffer.occupied);
}

void PacketNetwork::ready(uint64_t id, uint64_t bytes) {
    auto& flow = flows_.at(id);
    flow.ready_bytes += bytes;
    if (!flow.queued) {
        sources_[flow.source].ready.push_back(id);
        flow.queued = true;
        active_sources_.insert(flow.source);
    }
}

void PacketNetwork::submit(uint64_t id, int source, int destination, uint64_t bytes) {
    if (source < 0 || source >= nodes_ || destination < 0 || destination >= nodes_ ||
        !bytes || bytes % flit_bytes_ || flows_.count(id))
        throw std::invalid_argument("Invalid transfer endpoints, byte count, or active flow ID");
    if (dma_ && sources_[source].bandwidth <= 0)
        throw std::invalid_argument("DMA source has no HBM bandwidth");
    if (access_delay_ > std::numeric_limits<Tick>::max() - now_)
        throw std::overflow_error("HBM access tick overflow");
    flows_.emplace(id, Flow{source, route(source, destination), bytes, 0, bytes,
                            now_ + access_delay_, false});
    ++stats_.submitted;
    active_sources_.insert(source);
    if (dma_) sources_[source].dma_queue.push_back(id);
    else ready(id, bytes);
    // Submissions at one HYDRA timestamp are batched before advance arbitrates.
}

void PacketNetwork::pump() {
    for (auto position = active_sources_.begin(); position != active_sources_.end();) {
        const int node = *position;
        auto& source = sources_[node];
        if (dma_ && !source.dma_busy && !source.dma_queue.empty()) {
            // Round robin among ready flows; otherwise wait for earliest access.
            auto next = std::find_if(source.dma_queue.begin(), source.dma_queue.end(),
                [this](uint64_t id) { return flows_.at(id).ready_at <= now_; });
            if (next == source.dma_queue.end())
                next = std::min_element(source.dma_queue.begin(), source.dma_queue.end(),
                    [this](uint64_t a, uint64_t b) { return flows_.at(a).ready_at < flows_.at(b).ready_at; });
            const uint64_t id = *next;
            source.dma_queue.erase(next);
            auto& flow = flows_.at(id);
            const uint64_t bytes = std::min(dma_burst_, flow.dma_remaining);
            flow.dma_remaining -= bytes;
            const double cost = std::ceil(double(bytes) * 1e12 / source.bandwidth);
            if (!std::isfinite(cost) || cost >= double(std::numeric_limits<Tick>::max() / 2))
                throw std::overflow_error("HBM service tick overflow");
            const Tick service = std::max<Tick>(1, cost);
            schedule(std::max(now_, flow.ready_at) - now_ + service,
                     Kind::DmaDone, node, {id, bytes, 0});
            source.dma_busy = true;
        }
        while (!source.ready.empty()) {
            const uint64_t id = source.ready.front();
            auto& flow = flows_.at(id);
            const uint64_t bytes = std::min(quantum_, flow.ready_bytes);
            if (bytes > buffer_bytes_ - ports_[node].occupied) break;
            source.ready.pop_front();
            flow.queued = false;
            flow.ready_bytes -= bytes;
            reserve(node, bytes);
            ports_[node].queue.push_back({id, bytes, 0});
            if (!ports_[node].busy) active_ports_.insert(node);
            ++stats_.packets;
            if (flow.ready_bytes) ready(id, 0);
        }
        if (source.ready.empty() && (source.dma_busy || source.dma_queue.empty()))
            position = active_sources_.erase(position);
        else ++position;
    }
    if (contenders_.empty()) contenders_.resize(ports_.size());
    for (int target : targets_) contenders_[target].clear();
    targets_.clear();
    candidates_.assign(active_ports_.begin(), active_ports_.end());
    for (int index : candidates_) {
        auto& port = ports_[index];
        if (port.busy || port.queue.empty()) continue;
        const auto& packet = port.queue.front();
        const auto& path = flows_.at(packet.flow).path;
        if (packet.hop + 1 == path.size()) start_packet(index);
        else {
            const int target = path[packet.hop + 1];
            if (contenders_[target].empty()) targets_.push_back(target);
            contenders_[target].push_back(index);
        }
    }
    std::sort(targets_.begin(), targets_.end());
    for (int target : targets_) {
        const auto& candidates = contenders_[target];
        if (candidates.empty()) continue;
        auto& downstream = ports_[target];
        // Competing inputs share downstream buffer credits in round-robin order.
        // Fixed port iteration priority would otherwise starve longer routes.
        const size_t first = std::upper_bound(candidates.begin(), candidates.end(),
                                             downstream.last_grant) - candidates.begin();
        for (size_t offset = 0; offset < candidates.size(); ++offset) {
            const int index = candidates[(first + offset) % candidates.size()];
            const auto& packet = ports_[index].queue.front();
            if (packet.bytes > buffer_bytes_ - downstream.occupied) {
                ++stats_.blocked_checks;
                continue;
            }
            start_packet(index);
            downstream.last_grant = index;
        }
    }
}

void PacketNetwork::start_packet(int index) {
    auto& port = ports_[index];
    const Packet packet = port.queue.front();
    const auto& path = flows_.at(packet.flow).path;
    const bool last = packet.hop + 1 == path.size();
    int next = -1;
    if (!last) {
        next = path[packet.hop + 1];
        reserve(next, packet.bytes);
    }
    port.queue.pop_front();
    port.busy = true;
    active_ports_.erase(index);
    const uint64_t flits = packet.bytes / flit_bytes_;
    if (flits > (std::numeric_limits<Tick>::max() - hop_delay_) / period_)
        throw std::overflow_error("Packet serialization tick overflow");
    const Tick serialization = flits * period_;
    schedule(serialization, Kind::Depart, index, packet);
    if (last) schedule(serialization + hop_delay_, Kind::Deliver, index, packet);
    else {
        // Head arrival releases the next hop before the tail arrives.
        // Uniform link rates ensure the outgoing tail cannot overtake it.
        schedule(hop_delay_, Kind::Arrive, next,
                 {packet.flow, packet.bytes, packet.hop + 1});
    }
}

void PacketNetwork::process(const Event& event) {
    ++stats_.events;
    const auto& packet = event.packet;
    switch (event.kind) {
      case Kind::Arrive:
        ports_[event.port].queue.push_back(packet);
        if (!ports_[event.port].busy) active_ports_.insert(event.port);
        break;
      case Kind::Depart:
        ports_[event.port].busy = false;
        if (!ports_[event.port].queue.empty()) active_ports_.insert(event.port);
        ports_[event.port].occupied -= packet.bytes;
        break;
      case Kind::Deliver: {
        auto& flow = flows_.at(packet.flow);
        flow.outstanding -= packet.bytes;
        stats_.received_bytes += packet.bytes;
        if (!flow.outstanding) {
            completions_.push_back({packet.flow, now_});
            ++stats_.completed;
            flows_.erase(packet.flow);
        }
        break;
      }
      case Kind::DmaDone: {
        auto& source = sources_[event.port];
        source.dma_busy = false;
        ready(packet.flow, packet.bytes);
        if (flows_.at(packet.flow).dma_remaining)
            source.dma_queue.push_back(packet.flow);
        break;
      }
    }
}

std::vector<Completion> PacketNetwork::advance(Tick target) {
    if (target < now_) throw std::invalid_argument("Cannot reverse network time");
    completions_.clear();
    pump();
    while (!events_.empty() && events_.top().tick <= target) {
        now_ = events_.top().tick;
        // Equal-time arrivals/departures are visible before arbitration.
        do {
            const auto event = events_.top();
            events_.pop();
            process(event);
        } while (!events_.empty() && events_.top().tick == now_);
        pump();
        if (!completions_.empty()) return completions_;
    }
    if (events_.empty() && !flows_.empty())
        throw std::runtime_error("Packet network deadlock: active flows without events");
    now_ = target;
    return completions_;
}
} // namespace hydra
