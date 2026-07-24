import os
from typing import Tuple, List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import Sim.config.utils as utils
import Sim.common as common
from Sim.metrics.monitor import bandwidth_monitor

def are_adjacent(tile1: Tuple[int, int], tile2: Tuple[int, int]) -> bool:
    """Check if two tiles are 4-connected (adjacent)."""
    dx = abs(tile1[0] - tile2[0])
    dy = abs(tile1[1] - tile2[1])
    return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

class chip_graph:
    """
    A class representing a chip graph.
    """
    graph: nx.Graph
    num_nodes: int
    intp_width: int
    intp_height: int
    def __init__(self, num_nodes:int, intp_width:int, intp_height:int, chiplets_mapping:np.ndarray, two_d_grid:bool, placement_trace:list = None):
        self.num_nodes = num_nodes
        self.intp_width = intp_width
        self.intp_height = intp_height
        self.two_d_grid = two_d_grid
        self.id_to_coord = {}  # Maps node ID to coordinates (x, y)
        self.coord_to_id = {}
        
        if two_d_grid:
            self.graph = nx.grid_2d_graph(intp_width, intp_height)
            # Add chiplet type as a node attribute
            id = 0  # Initialize ID for nodes in the graph
            for x in range(intp_width):
                for y in range(intp_height):
                    self.graph.nodes[(x, y)]['chiplet_type'] = chiplets_mapping[x, y]
                    self.graph.nodes[(x, y)]['id'] = id # Each node represents a tile
                    id += 1
            self.id_to_coord = {data['id']: coord for coord, data in self.graph.nodes(data=True)}
            self.coord_to_id = {coord: data['id'] for coord, data in self.graph.nodes(data=True)}
            for u, v in self.graph.edges():
                self.graph.edges[u, v]['use'] = 0
                self.graph.edges[u, v]['bandwidth(GBs)'] = utils.NoI_bw  # Convert from Gbps to GB/s
                self.graph.edges[u, v]['latency(ns)'] = 10
                self.graph.edges[u,v]['bw_inuse'] = 0
        else:
            self.graph = nx.Graph()
            if placement_trace is None: # [[chiplet_type, [(tiles)]]]
                raise ValueError("Placement trace must be provided for non uniform grid graphs.")
            # Add one node per chiplet
            # Add each chiplet as one node
            id = 0
            for chiplet_id, (chiplet_type, tiles) in enumerate(placement_trace):
                self.graph.add_node(chiplet_id, chiplet_type=chiplet_type, coord=tiles, id = id)
                id += 1
            for i in range(len(placement_trace)):
                for j in range(i + 1, len(placement_trace)):
                    tiles_i = placement_trace[i][1]
                    tiles_j = placement_trace[j][1]
                    if any(are_adjacent(t1, t2) for t1 in tiles_i for t2 in tiles_j):
                        self.graph.add_edge(i, j)
                        self.graph.edges[i, j]['use'] = 0
                        self.graph.edges[i, j]['bandwidth(GBs)'] = utils.NoI_bw
                        self.graph.edges[i, j]['latency(ns)'] = 10
                        self.graph.edges[i, j]['bw_inuse'] = 0
                        
            self.num_nodes = len(placement_trace)
            self.id_to_coord = {id: data['coord'][0] for id, data in self.graph.nodes(data=True)}
            self.coord_to_id = {data['coord'][0]: id for id, data in self.graph.nodes(data=True)}
        bandwidth_monitor.init_monitor()
        # print("Bandwidth initialized with", utils.NoI_bw, "GB/s")

        # self.visualize()

    def visualize(self):
        nx.draw(self.graph, with_labels=True)
        output_file = os.path.join(common.output_folder, "chiplet_network.png")
        plt.savefig(output_file)
        plt.close()
    
    def is_2d_grid(self) -> bool:
        return self.two_d_grid

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def remove_edge(self, u, v):
        self.graph.remove_edge(u, v)

    def get_edges(self):
        return list(self.graph.edges())

    def __str__(self):
        return f"Chip Graph with {self.num_nodes} nodes and dimensions {self.intp_width}x{self.intp_height}"
    
    def find_unused_path(self, src_coord: tuple, dst_coord: tuple, size: float = 0):
        """
        Find a path from src_coord to dst_coord using only edges where 'use' < 4.
        Returns the list of (u, v) edges if such a path exists, or None otherwise.
        """
        visited = set()
        queue = deque([(src_coord, [])])  # (current_node, path_so_far)

        # TODO: use nx shortest_paths functions to first find all shortest paths 
        # If no shortest path found, then do current bfs approach to find any available path. 

        if self.is_2d_grid():
            while queue:
                current, path = queue.popleft()

                if current == dst_coord:
                    return path  # Reached destination with all 'use' == 0

                if current in visited:
                    continue
                visited.add(current)

                for neighbor in self.graph.neighbors(current):
                    if (current, neighbor) in path or (neighbor, current) in path:
                        continue  # prevent revisiting same edge in undirected graph
                    edge = self.graph.edges[current, neighbor]
                    if edge.get('use') < 4 and edge.get('bw_inuse', 0) + size <= edge.get('bandwidth(GBs)', utils.NoI_bw):
                        queue.append((neighbor, path + [(current, neighbor)]))
        else:
            while queue:
                current, path = queue.popleft()

                if current == dst_coord:
                    return path  # Reached destination with all 'use' == 0

                if current in visited:
                    continue
                visited.add(current)

                # convert current to node id if current is int and not tuple
                current_id = self.coord_to_id[current] if isinstance(current, tuple) else current

                for neighbor in self.graph.neighbors(current_id):
                    # convert current and neighbor to coords if they are int
                    neighbor_id = neighbor
                    neighbor = self.id_to_coord[neighbor_id] if isinstance(neighbor_id, int) else neighbor_id

                    if (current, neighbor) in path or (neighbor, current) in path:
                        continue  # prevent revisiting same edge in undirected graph
                    
                    edge = self.graph.edges[current_id, neighbor_id]
                    if edge.get('use') < 4 and edge.get('bw_inuse', 0) + size <= edge.get('bandwidth(GBs)', utils.NoI_bw):
                        queue.append((neighbor, path + [(current, neighbor)]))

        return None  # No valid path found

    def reserve_path(self, path: List[tuple], size: float = 0, timestep: int = 0) -> None:
        """
        Reserve a path by setting 'use' = 1 for each edge in the path.
        Assumes the path is a valid list of (u, v) tuples returned by find_unused_path().
        """
        for u, v in path:
            if self.is_2d_grid():
                u_id = u
                v_id = v
            else:
                # convert u and v to node ids if they are coords
                u_id = self.coord_to_id[u] if isinstance(u, tuple) else u
                v_id = self.coord_to_id[v] if isinstance(v, tuple) else v

            if self.graph.has_edge(u_id, v_id) and self.graph.edges[u_id, v_id]['use'] < 4 \
                and self.graph.edges[u_id, v_id]['bw_inuse'] + size <= self.graph.edges[u_id, v_id]['bandwidth(GBs)']:
                
                self.graph.edges[u_id, v_id]['use'] += 1
                self.graph.edges[u_id, v_id]['bw_inuse'] += size
                bandwidth_monitor.update_edge_bw(timestep=timestep, G=self.graph, u=u_id, v=v_id)
            else:
                raise ValueError(f"Edge ({u}, {v}) not found in the graph or in use.")

    def release_path(self, path: List[tuple], size: float = 0, timestep: int = 0):
        """
        Release a path by setting 'use' = 0 for each edge in the path.
        Assumes the path is a valid list of (u, v) tuples returned by find_unused_path().
        """
        for u, v in path:
            if self.is_2d_grid():
                u_id = u
                v_id = v
            else:
                # convert u and v to node ids if they are coords
                u_id = self.coord_to_id[u] if isinstance(u, tuple) else u
                v_id = self.coord_to_id[v] if isinstance(v, tuple) else v

            if self.graph.has_edge(u_id, v_id) and self.graph.edges[u_id, v_id]['use'] > 0:
                self.graph.edges[u_id, v_id]['use'] -= 1
                self.graph.edges[u_id, v_id]['bw_inuse'] -= size
                bandwidth_monitor.update_edge_bw(timestep=timestep, G=self.graph, u=u_id, v=v_id)
            else:
                raise ValueError(f"Edge ({u}, {v}) not found in the graph or not in use.")
    
    def get_distance(self, src_coord: tuple, dst_coord: tuple) -> float:
        """
        Calculate the distance between two coordinates in the chip graph.
        Uses the Euclidean distance formula.
        """
        # if src_coord not in self.id_to_coord.values() or dst_coord not in self.id_to_coord.values():
        #     raise ValueError("Coordinates not found in the chip graph.")
        
        x1, y1 = src_coord
        x2, y2 = dst_coord
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def check_bw_availability(self, chiplet_id, size: float) -> bool:
        """
        Check bandwidth availability for a given chiplet ID.
        """
        if self.is_2d_grid():
            c_id = self.id_to_coord[chiplet_id] if isinstance(chiplet_id, int) else chiplet_id
        else:
            c_id = chiplet_id

        if 'bw_inuse' not in self.graph.nodes[c_id]:
            raise ValueError(f"Chiplet {chiplet_id} does not have bandwidth attributes.")
        if self.graph.nodes[c_id]['bw_inuse'] + size > self.graph.nodes[c_id]['bandwidth']: # GB/s
            return False
        else:
            return True
    
    def load_bw_byid(self, chiplet_id: int, size: float, timestep: int) -> None:
        """
        Load bandwidth into a memory chiplet by its ID.
        """
        if self.is_2d_grid():
            c_id = self.id_to_coord[chiplet_id] if isinstance(chiplet_id, int) else chiplet_id
        else:
            c_id = chiplet_id

        if 'bw_inuse' not in self.graph.nodes[c_id]:
            raise ValueError(f"Chiplet {chiplet_id} does not have bandwidth attributes.")
        self.graph.nodes[c_id]['bw_inuse'] += size
        bandwidth_monitor.update_node_bw(timestep, self.graph, c_id)

    def offload_bw_byid(self, chiplet_id: int, size: float, timestep: int) -> None:
        """
        Offload bandwidth from a memory chiplet by its ID.
        """
        if self.is_2d_grid():
            c_id = self.id_to_coord[chiplet_id] if isinstance(chiplet_id, int) else chiplet_id
        else:
            c_id = chiplet_id
        
        if 'bw_inuse' not in self.graph.nodes[c_id]:
            raise ValueError(f"Chiplet {chiplet_id} does not have bandwidth attributes.")
        if self.graph.nodes[c_id]['bw_inuse'] < size:
            raise ValueError(f"Not enough bandwidth in chiplet {chiplet_id} to offload data.")
        self.graph.nodes[c_id]['bw_inuse'] -= size
        bandwidth_monitor.update_node_bw(timestep, self.graph, c_id)