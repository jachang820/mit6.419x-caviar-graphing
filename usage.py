import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

# Put the graphing module in the same directory.
import caviar

A = load_npz('cooffend_matrix.npz')
G = nx.from_scipy_sparse_matrix(A)
components = sorted(list(nx.connected_components(G)), 
  key=len, reverse=True)

# Plot the second largest component
component = components[1]
nodes = [(node, data) for node, data in G.nodes(data=True) 
        if node in component]
edges = [(node, neighbor, data)
        for node, neighbors in G.adj.items() if node in component
        for neighbor, data in neighbors.items() if neighbor in component]
SG = G.__class__()
SG.add_nodes_from(nodes)
SG.add_edges_from(edges)

G, fig, ax = caviar.plot(
  # Adjacency matrix or Networkx graph.
  # Accepted: (n x n) np.ndarray | (n x n) scipy.csr_matrix |
  #           nx.Graph | nx.DiGraph
  SG,
  
  # Treat adjacency matrix as a directed graph.
  # Accepted: True | False
  # Default: False (Undirected)
  directed=False,

  # Color scheme to set graph nodes and edges.
  # Accepted: 'teal' | 'wine' | 'red' | 'blue' | 'orange' | 'green' |
  #           'purple' | 'mint' | 'yellow' |
  #           tuple(edge_color, node_color, rng_start, rng_end)
  #           where:
  #               edge_color -- any acceptable MatPlotLib color
  #               node_color -- any acceptable MatPlotLib color, OR
  #                             a function that takes in a float between
  #                             [0., 1.] and maps it to a Matplotlib
  #                             color
  #               rng_start,
  #               rng_end    -- Narrow the range of values that the
  #                             function accepts. Ignored if node_color
  #                             is a single color.
  #                             
  #           For example, if 'c' is a float in [0., 1.], and node_color
  #           is a function, then the color is mapped as:
  #            
  #            color = node_color(c * (rng_end - rng_start) + rng_start)
  # Default: 'red'
  scheme='blue',

  # Score mechanism to focus on in the graph. Increases size of nodes or
  # adjusts the color if node_color is a function.
  # Accepted: 'eigenvector' | 'degree' | 'betweenness' | 'hub' |
  #           'authority' | 'clustering' | 'clustering_small' | function |
  #           None
  #           where:
  #               'clustering_small' -- (1 - clustering), treat coef 0.
  #                                     as most important nodes
  #               function           -- a function that takes in a 
  #                                     Networkx graph and returns a
  #                                     dict { node: numerical_score }
  # Default: 'eigenvector'
  context='betweenness',

  # Plot only nodes based on top score.
  # Accepted: int | float > 0
  #           where:
  #               Any negative int   -- use all nodes
  #               Any positive int   -- use top n scored nodes
  #               float in [0., 1.]  -- use ratio of nodes
  #               
  #           For example, if float, then the nodes kept are
  #            keep_top = int(keep_top * len(G.nodes))
  # Default: 100
  keep_top=-1,

  # Nodes that should be drawn and labeled regardless of score.
  # Accepted: None | list of nodes e.g. ['n1', 'n3', 'n12', 'n87']
  # Default: None
  must_keep=None,

  # Node size range.
  # Accepted: iterable (min_size, max_size)
  # Default: (40, 300)
  size_range=(40, 300),

  # Function to map a float in [0., 1.] to node size.
  # Accepted: 'log' | 'linear' | 'exp' | function
  #           where:
  #               'log'    -- log_2(1 + score)
  #               'linear' -- score
  #               'exp'    -- exp(score) - 1
  #               function -- a funcion that takes in a float in
  #                           [0., 1.] and maps it to a float in
  #                           [0., 1.]
  #           For example, given input 'score' and function f,
  #           node_size = f(score) * (max_size - min_size) + min_size
  # Default: 'log'
  size_function='log',

  # Node border width.
  # Accepted: Positive float
  # Default: 0.5
  border_width=0.5,

  # Edge width range.
  # Accepted: iterable (min_size, max_size)
  # Default: (0.4, 2.5)
  edge_width_range=(0.4, 2.),

  # Function to map a float in [0., 1.] to edge width.
  # Accepted: (same as size_function)
  # Default: 'exp'
  edge_width_function='exp',

  # The kind of value to label nodes with.
  # Accepted: 'name' | 'score' | 'rank'
  #           where:
  #               'name'  -- name/label of node
  #               'score' -- score of scoring mechanism
  #               'rank'  -- rank of scoring mechanism
  # Default: 'name'
  label_type='rank',

  # Label only nodes based on top score.
  # Accepted: (same as keep_top)
  #           For example, if label_top > keep_top, then all nodes
  #           are kept. If label_top is float and keep_top is int, 
  #           then the number of labeled nodes are
  #             label_top * keep_top
  # Default: 10
  label_top=10,

  # Font size of label.
  # Accepted: Positive int
  # Default: 10
  font_size=8,

  # Font weight of label.
  # Accepted: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 
  #           'ultralight'
  # Default: 'normal'
  font_weight='normal',

  # Proportion of nodes to randomly sample for betweenness centrality
  # in order to increase speed.
  # Accepted: Positive int (will be set to n if > n)
  # Default: 5000
  k=5000,

  # Random seed for sampling k with betweenness centrality.
  # Accepted: Positive int
  # Default: 42
  seed=42,

  # Whether to print updates to console for diagnostic purposes.
  # Accepted: True | False
  # Default: True (recommended for betweenness)
  verbose=True,

  # Axes object used by Matplotlib in case you want to add graph to an
  # existing figure. If None is passed, plot() will create one and
  # return it. Useful for creating multiple graphs per figure, or to
  # customize options.
  # Accepted: matplotlib.pyplot.axes
  # Default: None
  ax=None)

ax.set_title("Betweenness centrality\nsecond largest component")
plt.tight_layout()
plt.savefig('sample_graph.png', dpi=300, format='png')

plt.close()