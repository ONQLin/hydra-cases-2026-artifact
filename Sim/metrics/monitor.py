from dataclasses import dataclass, field
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.colors as pc

from typing import Any, Dict, Optional, List
from Sim.entities.mem_chiplet import mem_chiplet
from Sim.entities.comp_chiplet import comp_chiplet
from Sim.config.utils import chiplet_types_list, IO_bw
import Sim.config.utils as utils


@dataclass
class analytics:
    """
    A class to hold analytics number from the analytic model.
    """
    total_latency: int = field(
        default=0,
        metadata={"help": "Total latency in nanoseconds (1G clk)."},
    )
    
    mem_time: int = field(
        default=0,
        metadata={"help": "Total memory time in nanoseconds (1G clk)."},
    )
    
    bw_used: float = field(
        default=0.0,
        metadata={"help": "Total bandwidth used in bytes."},
    )

    tracking: List = field(
        default_factory=list,
        metadata={"help": "List of tracking information."},
    )
    
    utilization: int = field(
        default=0,
        metadata={"help": "Total utilization of the cluster given the kernel."},
    )
    
    power: int = field(
        default=0,
        metadata={"help": "Total power consumption in mW for the cluster."},
    )

@dataclass    
class request_counter:
    """
    A class to hold the request counter.
    """
    total_requests: int = 0
    completed_requests: int = 0
    pending_requests: int = 0
    running_requests: int = 0
    running_batches: int = 0
    
    @classmethod
    def increment_running_batches(cls):
        cls.running_batches += 1
    
    @classmethod
    def decrement_running_batches(cls):
        cls.running_batches -= 1
        if cls.running_batches < 0:
            raise ValueError("running_batches cannot be negative.")
    
    @classmethod
    def increment_total(cls):
        cls.total_requests += 1
    
    @classmethod
    def increment_completed(cls):
        cls.completed_requests += 1
    
    @classmethod
    def increment_pending(cls):
        cls.pending_requests += 1
    
    @classmethod
    def increment_running(cls):
        cls.running_requests += 1
    
    @classmethod
    def decrement_pending(cls):
        cls.pending_requests -= 1
    
    @classmethod
    def decrement_running(cls):
        cls.running_requests -= 1
    
    @classmethod
    def decrement_completed(cls):
        cls.completed_requests -= 1
    
    @classmethod
    def decrement_total(cls):
        cls.total_requests -= 1
    
    @classmethod
    def reset(cls):
        cls.total_requests = 0
        cls.completed_requests = 0
        cls.pending_requests = 0
        cls.running_requests = 0

@dataclass    
class tokens_monitor:
    output_tokens: int = 0
    finished_tokens: int = 0 # tokens that from finished requests
    time_to_first_token: List[int] = field(default_factory=list)
    time_per_output_token: List[int] = field(default_factory=list)

    @classmethod
    def reset(cls):
        cls.output_tokens = 0
        cls.finished_tokens = 0
        cls.time_to_first_token = []
        cls.time_per_output_token = []
        
    @classmethod
    def inc_output_tokens(cls):
        cls.output_tokens += 1
        
    @classmethod
    def dec_output_tokens(cls, num: int = 1):
        cls.output_tokens = max(0, cls.output_tokens - max(0, num))

    @classmethod
    def inc_finished_tokens(cls, num: int = 1):
        cls.finished_tokens += num

    @classmethod
    def add_first_token(cls, timestamp: int):
        cls.time_to_first_token.append(timestamp)
    
    @classmethod
    def add_time_dec_token(cls, timestamp: int):
        cls.time_per_output_token.append(timestamp)

@dataclass
class congestion_monitor:
    congested_status: Dict[int, int] = field(
        default_factory=lambda: {},
        metadata={"help": "Dictionary to hold congested status of each memory chiplet."},
    )
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the congested status.
        """
        cls.congested_status = {}
    
    @classmethod
    def total_congested(cls) -> int:
        """
        Return the total number of congested memory chiplets.
        """
        return sum(value for value in cls.congested_status.values())
    
    @classmethod
    def inc_congested(cls, batch_id: int) -> None:
        """
        Increment the congested count for a memory chiplet.
        """
        cls.congested_status[batch_id] = cls.congested_status.get(batch_id, 0) + 1
        
    @classmethod
    def clear_congested(cls, batch_id: int) -> None:
        """
        Clear the congested status.
        """
        # remove the item
        if batch_id in cls.congested_status:
            del cls.congested_status[batch_id]

@dataclass
class mem_monitor:
    """
    A class to monitor the memory system.
    """
    hbm_locs: list = field(
        default_factory=list,
        metadata={"help": "List of HBM locations."},
    )
    Total_utilization_traces: Dict[int, List[float]] = field(
        default_factory=lambda: {},
        metadata={"help": "Dictionary to hold utilization traces at different timestep. key = time step, value = utilization."},
    )

    Ind_Utilization_traces: list[Dict[int, List[float]]] = field(
        default_factory=list,
        metadata={"help": "List to hold individual utilization traces for each memory chiplet."},  
    )
    
    @classmethod
    def init_monitor(cls, mems: List[mem_chiplet]) -> None: 
        """
        Initialize the memory monitor with a list of memory systems.
        """

        cls.hbm_locs = [mem.chiplet_loc for mem in mems if ('HBM' in chiplet_types_list[mem.chiplet_type]) or
                                               ('GDDR' in chiplet_types_list[mem.chiplet_type])]
        cls.Ind_Utilization_traces = [{} for _ in range(len(mems))]
        cls.Total_utilization_traces = {}
        if len(cls.hbm_locs) != len(mems):
            raise ValueError("HBM locations do not match the number of memory systems.")
    
    @classmethod
    def update_utilization(cls, mems: List[mem_chiplet], total_util: float, time_step: int) -> None:
        """
        Update the utilization traces.
        """
        # check the total and seperate util
        if utils.verbose == True:
            if time_step not in cls.Total_utilization_traces:
                cls.Total_utilization_traces[time_step] = []
            cls.Total_utilization_traces[time_step].append(total_util)
            
            counter_mem = 0
            for idx, mem in enumerate(mems):
                if time_step not in cls.Ind_Utilization_traces[idx]:
                    cls.Ind_Utilization_traces[idx][time_step] = []
                cls.Ind_Utilization_traces[idx][time_step].append(mem.inuse_budget)
                counter_mem += mem.inuse_budget
            if abs(counter_mem - total_util) > 1e-2:  # Use a small epsilon for floating point comparison
                raise ValueError("Total utilization does not match the sum of individual utilizations.")
        
    @classmethod
    def visualize_trace(cls, total_budget, individual_budget=16*1024, output_dir="output/") -> None:
        """
        Visualize the utilization traces.
        """
        
        # configure font size
        plt.rcParams.update({'font.size': 16})
        # configure tick size
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16})
        
        # Plot total utilization
        plt.figure(figsize=(12, 6))
        time_steps = []
        Total_Usage = []
        for time_step, utilizations in cls.Total_utilization_traces.items():
            time_steps.append(time_step)
            Total_Usage.append(utilizations[-1])
        plt.plot(time_steps, Total_Usage, marker='o')
        plt.title("Total Memory Usage Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Mem Usage (MB)")
        plt.savefig(f"{output_dir}/total_memory_usage.png")
        # print("Total Memory Usage Over Time figure saved to output/total_memory_usage.png")

        time_steps = []
        Total_Utils = []
        # generate another figure
        plt.figure(figsize=(12, 6))
        for time_step, utilizations in cls.Total_utilization_traces.items():
            time_steps.append(time_step)
            Total_Utils.append((utilizations[-1]/total_budget) * 100)  # Convert to percentage
        plt.plot(time_steps, Total_Utils, marker='o', color='orange')
        plt.title("Total Memory Utilization Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Mem Usage (%)")
        plt.savefig(f"{output_dir}/total_memory_utilization.png")
        # print the average utilization
        avg_util = sum(Total_Utils) / len(Total_Utils)
        return avg_util
        
        # print("Total Memory Utilization Over Time figure saved to output/total_memory_utilization.png")



@dataclass
class comp_monitor:
    """
    A class to monitor the compute system.
    """
    comp_locs: list = field(
        default_factory=list,
        metadata={"help": "List of comp locations."},
    )
    Total_run_traces: Dict[int, List[float]] = field(
        default_factory=lambda: {},
        metadata={"help": "Dictionary to hold utilization traces at different timestep. key = time step, value = utilization."},
    )
    Ind_Utilization_traces: list[Dict[int, List[float]]] = field(
        default_factory=list,
        metadata={"help": "List to hold individual utilization traces for each comp chiplet."},  
    )
    Total_power_traces: Dict[int, List[float]] = field(
        default_factory=lambda: {},
        metadata={"help": "Dictionary to hold power traces at different timestep. key = time step, value = power."},
    )
    Ind_Power_traces: list[Dict[int, List[float]]] = field(
        default_factory=list,
        metadata={"help": "List to hold individual power traces for each compute chiplet."},
    )

    @classmethod
    def init_monitor(cls, comps: List[comp_chiplet]) -> None:
        """
        Initialize the compute monitor with the number of runs and power.
        """
        cls.comp_locs = [comp.chiplet_loc for comp in comps if '_' in chiplet_types_list[comp.chiplet_type]]
        cls.Ind_Utilization_traces = [{} for _ in range(len(comps))]
        cls.avg_util = 0.0
        cls.Total_run_traces = {}
        cls.Total_power_traces = {}
        cls.Ind_Power_traces = [{} for _ in range(len(comps))]
        if len(cls.comp_locs) != len(comps):
            raise ValueError("Comp locations do not match the number of compute systems.")
        total_power = 0
        for idx, comp in enumerate(comps):
            total_power += comp.power
            # timestep = 0
            cls.Ind_Power_traces[idx][0] = [comp.power]
            cls.Ind_Utilization_traces[idx][0] = [comp.util]
        cls.Total_power_traces[0] = [total_power]
        cls.avg_util = sum(comp.util for comp in comps) / len(comps)
        
    @classmethod
    def update_utilization(cls, comps: List[comp_chiplet], total_run: int, time_step: int) -> None:
        """
        Update the utilization traces.
        """
        if utils.verbose == True:
            if time_step not in cls.Total_run_traces:
                cls.Total_run_traces[time_step] = []
            cls.Total_run_traces[time_step].append(total_run)
            
            counter_comp = 0
            for idx, comp in enumerate(comps):
                if time_step not in cls.Ind_Utilization_traces[idx]:
                    cls.Ind_Utilization_traces[idx][time_step] = []
                cls.Ind_Utilization_traces[idx][time_step].append(comp.util)
                counter_comp += 1 if comp.idle != True else 0
            if counter_comp != total_run:
                raise ValueError("Total run does not match the sum of individual runs.")
        
    @classmethod
    def visualize_trace(cls, output_dir='output') -> None:
        """
        Visualize the memory utilization traces.
        """
        # configure font size
        plt.rcParams.update({'font.size': 16})
        # configure tick size
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16})
        
        time_steps = []
        Total_Runs = []
        # generate another figure
        plt.figure(figsize=(12, 6))
        for time_step, runs in cls.Total_run_traces.items():
            time_steps.append(time_step)
            Total_Runs.append(runs[-1])
        plt.plot(time_steps, Total_Runs, marker='o', color='green')
        plt.title("Total Comp Chips Runs Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Runs")
        plt.savefig(f"{output_dir}/total_comp_chips_runs.png")
        # print("Total Comp Chips Runs Over Time figure saved to output/total_comp_chips_runs.png")
        
        time_steps = []
        avg_utils = []
        plt.figure(figsize=(12, 6))
        for time_step, utilizations in cls.Ind_Utilization_traces[0].items():
            time_steps.append(time_step)
            util = 0
            for idx in range(len(cls.Ind_Utilization_traces)):
                util += cls.Ind_Utilization_traces[idx][time_step][-1]
            avg_utils.append(util / len(cls.Ind_Utilization_traces))

        plt.plot(time_steps, avg_utils, marker='o')
        plt.title("Total Comp Chiplet Utilization Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Utilization (%)")
        plt.savefig(f"{output_dir}/total_comp_chips_utilization.png")
        
        # return avg utilization
        if len(time_steps) > 1:
            weighted_sum = 0.0
            total_time = 0.0
            for i in range(len(time_steps) - 1):
                dt = time_steps[i + 1] - time_steps[i]
                weighted_sum += Total_Runs[i] * dt
                total_time += dt
            avg_util = weighted_sum / total_time if total_time > 0 else 0.0
        else:
            # If only one timestep, use simple average
            avg_util = sum(Total_Runs) / len(Total_Runs) if Total_Runs else 0.0
        
        return avg_util
        # print("Total Comp Chiplet Utilization Over Time figure saved to output/total_comp_chips_utilization.png")

@dataclass
class bandwidth_monitor:
    """
    Monitor class to collect traces of bandwidth usage for nodes and edges
    in a chiplet interconnect graph.
    """
    
    node_traces: List[Dict[str, Any]] = field(default_factory=list,
    metadata={"help": "List of chiplet IO bandwidth usage traces{'timestep','id', 'chiplet_type', 'bandwidth(GBs), 'bw_inuse', 'loc'}."})

    edge_traces: List[Dict[str, Any]] = field(default_factory=list,
    metadata={"help": "List of chiplet interconnect edge bandwidth usage traces{'timestep','u','v','bandwidth(GBs)','bw_inuse'}."})

    @classmethod
    def init_monitor(cls):
        cls.node_traces = []
        cls.edge_traces = []

    @classmethod
    def update_node_bw(cls, timestep: int, G: nx.Graph, coord: tuple) -> None:
        """
        Record a single node's bandwidth usage trace from the graph,
        where the node key is its 2D coordinate.
        """
        if utils.verbose == True:
            data: Dict[str, Any] = G.nodes[coord]
            cap = data.get("bandwidth")
            use = data.get("bw_inuse")
            chiplet_type = data.get("chiplet_type")
            if cap is None or cap == 0:
                raise ValueError(f"Invalid bandwidth {cap} for node {coord}")
            util = use / cap
            chip_id = data.get("id")
            if util < 0 or util > 1:
                raise ValueError(f"Invalid utilization {util} for node {coord}")
            cls.node_traces.append({
                "timestep": timestep,
                "id": chip_id,                 # coordinate is the ID
                "chiplet_type": chiplet_type,
                "bandwidth_GBs": cap,
                "bw_inuse": use,
                "utilization": util,
                "loc": coord,                # loc = coord in this grid model
            })

    @classmethod
    def update_edge_bw(cls, timestep: int, G: nx.Graph, u: tuple, v: tuple) -> None:
        """
        Record a single edge's bandwidth usage trace from the graph,
        where the edge is identified by its two endpoint coordinates.
        """
        if utils.verbose == True:
            data: Dict[str, Any] = G.edges[u, v]   # <-- NetworkX edge dict access
            cap = data.get("bandwidth(GBs)")
            use = data.get("bw_inuse")

            if cap is None or cap <= 0:
                raise ValueError(f"Invalid bandwidth {cap} for edge ({u}, {v})")

            util = use / cap
            if util < 0 or util > 1:
                raise ValueError(f"Invalid utilization {util:.3f} for edge ({u}, {v})")

            cls.edge_traces.append({
                "timestep": timestep,
                "u": u,
                "v": v,
                "bandwidth_GBs": cap,
                "bw_inuse": use,
                "utilization": util,
            })
    
    @classmethod
    def visualize_trace(cls, output_dir='output') -> None:
        
        nodes_df = pd.DataFrame(cls.node_traces).sort_values(["id","timestep"])
        fig = go.Figure()

        # --- Edges ---
        edges_df = pd.DataFrame(cls.edge_traces).sort_values(["u","v","timestep"])
        edges_df["dt"] = edges_df.groupby(["u","v"])["timestep"].diff().shift(-1).fillna(0)
        weighted_util_e = (edges_df["utilization"] * edges_df["dt"]).groupby([edges_df["u"], edges_df["v"]]).sum()
        total_time_e = edges_df.groupby(["u","v"])["dt"].sum()
        avg_util_e = (weighted_util_e / total_time_e).reset_index(name="avg_utilization")

        # build node location map from raw traces, not aggregated avg_util
        loc_map = nodes_df.groupby("id")["loc"].first().to_dict()

        def util_to_color(val): 
            val = max(0.0, min(1.0, val))  
            return px.colors.sample_colorscale("Reds", [val])[0]  # white → deep red

        for _, row in avg_util_e.iterrows():
            u, v = row["u"], row["v"]
            util = row["avg_utilization"]
            if pd.isna(util):
                util = 0.0

            x0, y0 = u
            x1, y1 = v
            color = util_to_color(util)

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(
                    color=color,
                    width=2 + 12*util
                ),
                # send edge info into customdata
                customdata=[[u, v, util]] * 2,  # two points (x0,y0) and (x1,y1)
                hovertemplate=(
                    "Edge: %{customdata[0]} → %{customdata[1]}<br>"
                    "Avg Util: %{customdata[2]:.2f}<extra></extra>"
                ),
                showlegend=False
            ))
            
            # invisible hover marker at midpoint
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[xm], y=[ym],
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)"),  # invisible
                customdata=[[u, v, util]],
                hovertemplate=(
                    "Edge: %{customdata[0]} → %{customdata[1]}<br>"
                    "Avg Link Util: %{customdata[2]:.2f}<extra></extra>"
                ),
                showlegend=False
            ))

        # --- Nodes ---
        nodes_df["dt"] = nodes_df.groupby("id")["timestep"].diff().shift(-1).fillna(0)
        weighted_util = (nodes_df["utilization"] * nodes_df["dt"]).groupby(nodes_df["id"]).sum()
        total_time = nodes_df.groupby("id")["dt"].sum()
        avg_util = (weighted_util / total_time).reset_index(name="avg_utilization")

        locs = nodes_df.groupby("id")["loc"].first().reset_index()
        types = nodes_df.groupby("id")["chiplet_type"].first().reset_index()
        avg_util = avg_util.merge(locs, on="id", how="left")
        avg_util = avg_util.merge(types, on="id", how="left")
        avg_util[["x","y"]] = pd.DataFrame(avg_util["loc"].tolist(), index=avg_util.index)
        label_dict = {
            0: "Mp", 1: "Md", 2: "Ap", 3: "Ad", 5: "Mem"
        }
        avg_util["label"] = avg_util["chiplet_type"].apply(lambda t: label_dict[t] if t in label_dict else "X")

        fig.add_trace(go.Scatter(
            x=avg_util["x"],
            y=avg_util["y"],
            mode="markers+text",
            text=avg_util["label"],  
            textfont=dict(
                size=16,       # increase font size
                color="white", # text color
                family="Arial", 
                # weight="bold"   # <- not supported directly
            ),
            textposition="middle center",
            marker=dict(
                size=60,
                color=avg_util["avg_utilization"],
                colorscale=[[0, "rgb(255,200,200)"], [1, "rgb(180,0,0)"]],  # light red → dark red
                opacity=1,   # <-- let edges show through
                cmin=0, cmax=1,
                line=dict(color="black", width=2)
            ),
            hovertemplate=(
                "Label: %{text}<br>"
                "Chiplet Type: %{customdata[0]}<br>"
                "Coord: %{x}, %{y}<br>"
                "Avg Util: %{marker.color:.2f}<extra></extra>"
            ),
            customdata=avg_util[["chiplet_type"]].values,
        ))
        
              # Save aggregated utilizations to JSON
        os.makedirs(output_dir, exist_ok=True)
        util_snapshot = {
            "nodes": [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "chiplet_type": int(row["chiplet_type"]),
                    "coord": [float(row["x"]), float(row["y"])],
                    "avg_utilization": float(row["avg_utilization"])
                }
                for _, row in avg_util.iterrows()
            ],
            "edges": [
                {
                    "u": list(map(float, row["u"])),
                    "v": list(map(float, row["v"])),
                    "avg_utilization": float(row["avg_utilization"])
                }
                for _, row in avg_util_e.iterrows()
            ],
        }
        with open(os.path.join(output_dir, "chipletBW_utilization.json"), "w") as f:
            json.dump(util_snapshot, f, indent=2)
        
        fig.update_layout(
            title="Chiplet Bandwidth Utilization Heatmap",
            xaxis=dict(title="X", scaleanchor="y", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            plot_bgcolor="white",
        )

        fig.write_html(f"{output_dir}/chipletBW_utilization_heatmap.html")
        
    @classmethod
    def visualize_IO_trace(cls, output_dir='output') -> None:
        nodes_df = pd.DataFrame(cls.node_traces).sort_values(["id","timestep"])
        edges_df = pd.DataFrame(cls.edge_traces).sort_values(["u","v","timestep"])

        # --------- NODES (IO bandwidth) ---------
        nodes_df["dt"] = nodes_df.groupby("id")["timestep"].diff().shift(-1).fillna(0)
        weighted_util = (nodes_df["utilization"] * nodes_df["dt"]).groupby(nodes_df["id"]).sum()
        total_time = nodes_df.groupby("id")["dt"].sum()
        avg_util_nodes = (weighted_util / total_time).reset_index(name="avg_utilization")

        locs = nodes_df.groupby("id")["loc"].first().reset_index()
        types = nodes_df.groupby("id")["chiplet_type"].first().reset_index()
        avg_util_nodes = avg_util_nodes.merge(locs, on="id", how="left")
        avg_util_nodes = avg_util_nodes.merge(types, on="id", how="left")
        avg_util_nodes[["x","y"]] = pd.DataFrame(avg_util_nodes["loc"].tolist(), index=avg_util_nodes.index)
        label_dict = {
            0: "Mp", 1: "Md", 2: "Ap", 3: "Ad", 5: "Mem"
        }
        avg_util_nodes["label"] = avg_util_nodes["chiplet_type"].apply(lambda t: label_dict[t] if t in label_dict else "X")

        fig_nodes = go.Figure()

        # --------- Dump Interconnect (just black lines) ---------
        # take unique edges
        unique_edges = edges_df.groupby(["u","v"]).size().reset_index()[["u","v"]]
        for _, row in unique_edges.iterrows():
            u, v = row["u"], row["v"]   # already coordinate tuples
            x0, y0 = u
            x1, y1 = v
            fig_nodes.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",   # no hover text for dump lines
                showlegend=False
            ))

        # --------- Nodes ---------
        fig_nodes.add_trace(go.Scatter(
            x=avg_util_nodes["x"],
            y=avg_util_nodes["y"],
            mode="markers+text",
            text=avg_util_nodes["label"],
            textfont=dict(size=20, color="white", family="Arial"),
            textposition="middle center",
            marker=dict(
                size=60,
                color=avg_util_nodes["avg_utilization"],
                colorscale=[[0, "rgb(255,200,200)"], [1, "rgb(180,0,0)"]],
                cmin=0, cmax=1,
                line=dict(color="black", width=2)
            ),
            hovertemplate=(
                "Node Label: %{text}<br>"
                "Chiplet Type: %{customdata[0]}<br>"
                "Coord: %{x}, %{y}<br>"
                "Avg IO Util: %{marker.color:.2f}<extra></extra>"
            ),
            customdata=avg_util_nodes[["chiplet_type"]].values,
        ))

        fig_nodes.update_layout(
            title="Chiplet IO Bandwidth Utilization (with Interconnect)",
            xaxis=dict(title="X", scaleanchor="y", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            plot_bgcolor="white",
        )

        fig_nodes.write_html(f"{output_dir}/chipletIO_utilization.html")
        
    @classmethod
    def visualize_Link_trace(cls, output_dir='output') -> None:
        # --------- EDGES (Link bandwidth) ---------
        nodes_df = pd.DataFrame(cls.node_traces).sort_values(["id","timestep"])
        edges_df = pd.DataFrame(cls.edge_traces).sort_values(["u","v","timestep"])
        
        edges_df["dt"] = edges_df.groupby(["u","v"])["timestep"].diff().shift(-1).fillna(0)
        weighted_util_e = (edges_df["utilization"] * edges_df["dt"]).groupby([edges_df["u"], edges_df["v"]]).sum()
        total_time_e = edges_df.groupby(["u","v"])["dt"].sum()
        avg_util_edges = (weighted_util_e / total_time_e).reset_index(name="avg_utilization")

        fig_edges = go.Figure()

        # --- edges colored by utilization ---
        def util_to_color(val):
            val = max(0.0, min(1.0, val))
            return px.colors.sample_colorscale("Reds", [val])[0]

        for _, row in avg_util_edges.iterrows():
            u, v = row["u"], row["v"]   # already coords
            util = row["avg_utilization"]
            x0, y0 = u
            x1, y1 = v
            color = util_to_color(util)

            fig_edges.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=2 + 12*util),
                showlegend=False
            ))
            # invisible hover marker at midpoint
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            fig_edges.add_trace(go.Scatter(
                x=[xm], y=[ym],
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)"),  # invisible
                customdata=[[u, v, util]],
                hovertemplate=(
                    "Edge: %{customdata[0]} → %{customdata[1]}<br>"
                    "Avg Link Util: %{customdata[2]:.2f}<extra></extra>"
                ),
                showlegend=False
            ))

        # --- dummy nodes (neutral markers, no utilization) ---
        locs = nodes_df.groupby("id")["loc"].first().reset_index()
        locs[["x","y"]] = pd.DataFrame(locs["loc"].tolist(), index=locs.index)

        fig_edges.add_trace(go.Scatter(
            x=locs["x"],
            y=locs["y"],
            mode="markers+text",
            text=locs["id"],   # could also use chiplet_type or leave blank
            textposition="middle center",
            textfont=dict(size=12, color="black"),
            marker=dict(
                size=30,
                color="lightgrey",
                line=dict(color="black", width=1)
            ),
            hoverinfo="skip",   # dummy nodes don't need info
            showlegend=False
        ))

        fig_edges.update_layout(
            title="Chiplet Interconnect Bandwidth Utilization",
            xaxis=dict(title="X", scaleanchor="y", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            plot_bgcolor="white",
        )
        fig_edges.write_html(f"{output_dir}/chipletLink_utilization.html")
