import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ---------------- Graph Setup ----------------
graph_edges = [
    ("A", "B", 4), ("A", "C", 5),
    ("B", "C", 11), ("B", "D", 9), ("B", "E", 7),
    ("C", "E", 3),
    ("D", "F", 2), ("D", "E", 13),
    ("E", "F", 6)
]

G = nx.Graph()
G.add_weighted_edges_from(graph_edges)

pos = nx.spring_layout(G, seed=42)  # fixed layout for reproducibility

# ---------------- Dijkstra Algorithm ----------------
class DijkstraSimulator:
    def __init__(self, graph, source="A"):
        self.graph = graph
        self.source = source
        self.distances = {node: math.inf for node in graph.nodes}
        self.distances[source] = 0
        self.visited = set()
        self.path = {node: None for node in graph.nodes}
        self.steps = []
        self._prepare_steps()

    def _prepare_steps(self):
        unvisited = set(self.graph.nodes)
        while unvisited:
            # Select unvisited node with smallest distance
            current = min(unvisited, key=lambda node: self.distances[node])
            if self.distances[current] == math.inf:
                break
            self.visited.add(current)
            unvisited.remove(current)

            # Save step
            self.steps.append((current, dict(self.distances)))

            for neighbor in self.graph.neighbors(current):
                weight = self.graph[current][neighbor]["weight"]
                if neighbor not in self.visited:
                    new_dist = self.distances[current] + weight
                    if new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.path[neighbor] = current

    def get_steps(self):
        return self.steps

    def get_final_paths(self):
        return self.path, self.distances

# ---------------- GUI ----------------
class DijkstraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijkstra Algorithm Simulator")

        # Simulator
        self.sim = DijkstraSimulator(G, source="A")
        self.steps = self.sim.get_steps()
        self.current_step = -1

        # Graph display
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=6)

        # Distance table
        self.tree = ttk.Treeview(root, columns=("Node", "Distance"), show="headings")
        self.tree.heading("Node", text="Node")
        self.tree.heading("Distance", text="Distance")
        self.tree.grid(row=0, column=1, padx=10, pady=10)

        # Buttons
        ttk.Button(root, text="Next Step", command=self.next_step).grid(row=1, column=1, pady=5)
        ttk.Button(root, text="Show Result", command=self.show_result).grid(row=2, column=1, pady=5)
        ttk.Button(root, text="Reset", command=self.reset).grid(row=3, column=1, pady=5)

        # Initial graph draw
        self.draw_graph()

    def draw_graph(self, highlight_node=None):
        self.ax.clear()
        nx.draw(G, pos, with_labels=True, node_color="skyblue", ax=self.ax, node_size=1000)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), ax=self.ax)
        if highlight_node:
            nx.draw_networkx_nodes(G, pos, nodelist=[highlight_node], node_color="yellow", ax=self.ax, node_size=1000)
        self.canvas.draw()

    def next_step(self):
        if self.current_step + 1 < len(self.steps):
            self.current_step += 1
            node, distances = self.steps[self.current_step]
            self.draw_graph(highlight_node=node)
            self.update_table(distances)
        else:
            messagebox.showinfo("Done", "All steps completed!")

    def update_table(self, distances):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for node, dist in distances.items():
            self.tree.insert("", "end", values=(node, "∞" if dist == math.inf else dist))

    def show_result(self):
        path, distances = self.sim.get_final_paths()
        result = "\n".join([f"{node}: {distances[node]} (via {path[node]})" for node in distances])
        messagebox.showinfo("Final Shortest Distances", result)

    def reset(self):
        self.sim = DijkstraSimulator(G, source="A")
        self.steps = self.sim.get_steps()
        self.current_step = -1
        self.tree.delete(*self.tree.get_children())
        self.draw_graph()

# ---------------- Run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DijkstraGUI(root)
    root.mainloop()
