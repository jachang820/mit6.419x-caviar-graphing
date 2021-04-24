import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


first_time = True

dark_set3 = []
for r, g, b in cm.Set3.colors:
  dark_set3.append((
    np.round(0.9 * r, 2), 
    np.round(0.9 * g, 2), 
    np.round(0.9 * b, 2)))

print_verbose = lambda x: print("Not implemented.")


class ColorScheme:
  def __init__(self, edge, node, range_start=0.0, range_end=1.0):
    self.edge = edge
    self.node = node
    self.start = range_start
    self.end = range_end


  def edge_color(self):
    return self.edge


  def node_color(self, value=1.0):
    if not callable(self.node):
      return self.node
    else:
      return self.node(value * (self.end - self.start) + self.start)



schemes = {
  'teal': ColorScheme(dark_set3[0], cm.BuGn, 0.3, 0.65),
  'wine': ColorScheme(dark_set3[2], cm.pink_r, 0.45, 0.75),
  'red': ColorScheme(dark_set3[3], cm.RdPu, 0.3, 0.55),
  'blue': ColorScheme(dark_set3[4], cm.Blues, 0.3, 0.6),
  'orange': ColorScheme(dark_set3[5], cm.Oranges, 0.3, 0.65),
  'green': ColorScheme(dark_set3[6], cm.YlGn, 0.3, 0.55),
  'purple': ColorScheme(dark_set3[9], cm.BuPu, 0.4, 0.65),
  'mint': ColorScheme(dark_set3[10], cm.GnBu, 0.3, 0.5),
  'yellow': ColorScheme(dark_set3[11], cm.YlOrBr, 0.2, 0.5)
}



class NodeProperties:
  def __init__(self, G, keep_top, must_keep):
    sort_by = lambda el: el[1]['score'] 
    data = sorted(G.nodes(data=True), key=sort_by)
    keep_top = n_top_nodes(keep_top, len(data))
    important_data = self._important_nodes(data, keep_top, must_keep)
    data = self._get_top_data(data, important_data, keep_top)
    min_score, max_score = data[0][1]['score'], data[-1][1]['score']
    self.no_context = (min_score == max_score)
    if self.no_context:
      min_score, max_score = 0., 1.
    normalize = lambda score: (score - min_score) / (max_score - min_score)
    self.node_list = [node for node, d in data]
    self._get_pos(G, self.node_list)
    self.scores = {node: normalize(d['score']) for node, d in data}
    self.colors = None
    self.sizes = None
    self.alphas = None
    self.border = None
    self.linewidths = None


  def _important_nodes(self, data, keep_top, must_keep):
    important_data = []
    if must_keep is not None:
      must_keep = set(must_keep)
      for node_data in data[:keep_top]:
        node = node_data[0]
        if node in must_keep:
          important_data.append(node_data)
    return important_data


  def _get_top_data(self, data, important_data, keep_top):
    top_data = data[-keep_top:]
    data = important_data
    data.extend(top_data)
    return data


  def _get_pos(self, G, nodelist):
    node_set = set(self.node_list)
    remove_nodes = set(G.nodes) - node_set
    H = G.copy()
    H.remove_nodes_from(remove_nodes)
    self.pos = nx.drawing.nx_agraph.graphviz_layout(H)


  def set_color(self, scheme):
    if type(scheme) is tuple:
      scheme = ColorScheme(*scheme)
    else:
      scheme = schemes[scheme]
    self.colors = []
    for node in self.node_list:
      score = self.scores[node]
      r, g, b, a = scheme.node_color(score)
      color = (np.round(r, 2), np.round(g, 2), np.round(b, 2), a)
      self.colors.append(color)


  def set_size(self, size_range, size_function):
    self.sizes = []
    map_func = map_to_range_func(size_range, size_function)
    for node in self.node_list:
      score = self.scores[node]
      self.sizes.append(int(map_func(score)))


  def set_alpha(self):
    self.alphas = []
    map_func = map_to_range_func([0.4, 0.8], 'log')
    for node in self.node_list:
      score = self.scores[node]
      alpha = 0.8 if self.no_context else np.round(map_func(score), 2)
      self.alphas.append(alpha)


  def set_border(self, linewidths):
    self.linewidths = linewidths
    self.border = []
    map_func = map_to_range_func([0.4, 0.8], 'log')
    for node in self.node_list:
      score = self.scores[node]
      alpha = np.round(map_func(score), 2)
      self.border.append((0.4, 0.3, 0.4, alpha))


  @property
  def nodelist(self):
    return self.node_list


  @property
  def positions(self):
    return self.pos
  

  def properties(self, ax=None):
    props = {
      'node_color': self.colors,
      'node_size': self.sizes,
      'alpha': self.alphas,
      'edgecolors': self.border,
      'linewidths': self.linewidths,
      'nodelist': self.nodelist,
      'ax': ax
    }
    return {name: values for name, values in props.items()
      if values is not None}



class EdgeProperties:
  def __init__(self, G, nodelist):
    nodelist = set(nodelist)
    self.directed = nx.is_directed(G)
    get_weight = lambda el: el[2]['weight']
    max_weight = max(G.edges(data=True), key=get_weight)[2]['weight']
    min_weight = min(G.edges(data=True), key=get_weight)[2]['weight']
    normalize = lambda weight: (weight - min_weight) / (max_weight - min_weight)
    self.weights = {}
    seen_edges = set()
    for node, neighbors in G.adj.items():
      if node in nodelist:
        for neighbor, d in neighbors.items():
          if neighbor in nodelist:
            if self.directed or (neighbor, node) not in seen_edges:
              if 'weight' in d:
                self.weights[(node, neighbor)] = normalize(d['weight'])
              else:
                self.weights[(node, neighbor)] = 0
              seen_edges.add((node, neighbor))
    self.edge_list = list(self.weights.keys())
    self.colors = None
    self.widths = None
    self.arrowstyle = None


  def set_color(self, scheme):
    if type(scheme) is tuple:
      scheme = ColorScheme(*scheme)
    else:
      scheme = schemes[scheme]
    self.colors = [scheme.edge_color()] * len(self.edge_list)


  def set_width(self, edge_width_range, edge_width_function):
    self.widths = []
    map_func = map_to_range_func(edge_width_range, edge_width_function)
    for edge in self.edge_list:
      weight = self.weights[edge]
      self.widths.append(np.round(map_func(weight), 2))


  def set_arrowstyle(self):
    self.arrowstyle = '-|>' if self.directed else '-'


  @property
  def edgelist(self):
    return self.edge_list


  def properties(self, ax=None):
    props = {
      'edge_color': self.colors,
      'edgelist': self.edgelist,
      'width': self.widths,
      'arrowstyle': self.arrowstyle,
      'alpha': 0.8,
      'connectionstyle': 'arc3,rad=0.2',
      'ax': ax
    }
    return {name: values for name, values in props.items()
      if values is not None}



class LabelProperties:
  def __init__(self, G, nodelist, label_top):
    nodelist = set(nodelist)
    sort_by = lambda el: el[1]
    scores = [(node, data['score']) 
      for node, data in G.nodes(data=True) 
      if node in nodelist]
    scores = list(sorted(scores, key=sort_by))
    max_score = scores[0][1]
    self.score_exp = (max_score < 0.01)
    label_top = n_top_nodes(label_top, len(scores))
    self.scores = scores[-label_top:]
    self.labels = None
    self.font_size = None
    self.font_weight = None


  def set_labels(self, label_type):
    scores = self.scores
    if label_type == 'name':
      self.labels = {node: node for node, score in scores}
    elif label_type == 'score':
      if self.score_exp:
        score_format = lambda score: "{:.2e}".format(score)
      else:
        score_format = lambda score: np.round(score, 3)
      self.labels = {node: score_format(score) for node, score in scores}
    elif label_type == 'rank':
      self.labels = {node: i + 1 
        for i, (node, score) in enumerate(reversed(scores))}


  def set_style(self, font_size=None, font_weight=None):
    self.font_size = font_size
    self.font_weight = font_weight
    

  def properties(self, ax=None):
    props = {
      'labels': self.labels,
      'font_size': self.font_size,
      'font_weight': self.font_weight,
      'alpha': 0.75,
      'ax': ax
    }
    return {name: values for name, values in props.items()
      if values is not None}



def n_top_nodes(keep, n):
  if keep < 0:
    keep = n
  elif type(keep) is int:
    if keep > n:
      keep = n
  elif type(keep) is float:
    if keep > 1.0:
      keep -= int(keep)
    keep = int(keep * n)
  return keep



def map_to_range_func(dest_range, func):
  min_size, max_size = dest_range
  size_diff = max_size - min_size
  if callable(func):
    map_func = lambda score: func(score)
  elif func == 'log':
    map_func = lambda score: np.log2(1 + score)
  elif func == 'linear':
    map_func = lambda score: score
  elif func == 'exp':
    norm = np.exp(1) - 1
    map_func = lambda score: (np.exp(score) - 1) / norm
  map_to_dest = lambda score: map_func(score) * size_diff + min_size
  return map_to_dest



def plot(A, 
  directed=False,
  scheme='red',
  context='eigenvector',
  keep_top=100,
  must_keep=None,
  size_range=(40, 300),
  size_function='log',
  border_width=0.5,
  edge_width_range=(0.4, 2.5),
  edge_width_function='exp',
  label_type='name',
  label_top=10,
  font_size=10,
  font_weight='normal',
  k=5000,
  seed=42,
  verbose=True,
  ax=None):

  if seed is None:
    seed = np.random.default_rng().integers(1 << 16)

  global print_verbose
  print_verbose = verbose_func(verbose)

  adj_type = A.__class__.__name__
  if adj_type == 'Graph' or adj_type == 'DiGraph':
    G = A
  else:
    print_verbose("Creating graph from adjacency matrix...")
    G = create_graph(A, directed)

  print_verbose("Setting the score...")
  G = set_score(G, context, k, seed)

  print_verbose("Setting node properties...")
  node_props = NodeProperties(G, keep_top, must_keep)
  node_props.set_color(scheme)
  node_props.set_size(size_range, size_function)
  node_props.set_alpha()
  node_props.set_border(border_width)

  nodelist = node_props.nodelist
  pos = node_props.positions

  print_verbose("Setting edge properties...")
  edge_props = EdgeProperties(G, nodelist)
  edge_props.set_color(scheme)
  edge_props.set_width(edge_width_range, edge_width_function)
  edge_props.set_arrowstyle()

  print_verbose("Setting label properties...")
  label_props = LabelProperties(G, nodelist, label_top)
  label_props.set_labels(label_type)
  label_props.set_style(font_size, font_weight)

  print_verbose("Plotting graph...")
  if ax is None:
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()

  H = nx.DiGraph(G)
  nx.draw_networkx_nodes(H, pos=pos, **node_props.properties(ax))
  nx.draw_networkx_edges(H, pos=pos, **edge_props.properties(ax))
  if context is not None:
    nx.draw_networkx_labels(H, pos=pos, **label_props.properties(ax))
  
  return G, fig, ax


def create_graph(A, directed):
  graph_type = nx.DiGraph if directed else nx.Graph
  adj_type = A.__class__.__name__
  if adj_type == 'ndarray':
    G = nx.from_numpy_array(A, create_using=graph_type)
  elif adj_type == 'csr_matrix':
    G = nx.from_scipy_sparse_matrix(A, create_using=graph_type)
  return G


def set_score(G, context, k, seed):
  if callable(context):
    score = context(G)  
  elif context == 'eigenvector':
    score = nx.eigenvector_centrality(G, max_iter=1000000)
  elif context == 'degree':
    score = nx.degree_centrality(G)
  elif context == 'betweenness':
    score = betweenness_by_component(G, k, seed)
  elif context == 'hub':
    H = nx.DiGraph(G)
    score = nx.hits(max_iter=1000000)[0]
  elif context ==  'authority':
    H = nx.DiGraph(G)
    score = nx.hits(max_iter=1000000)[1]
  elif context == 'clustering':
    score = nx.clustering(G)
  elif context == 'clustering_small':
    score = nx.clustering(G)
    for node in score.keys():
      score[node] = 1 - score[node]
  elif context is None:
    score = {node: 0. for node in G.nodes}

  nx.set_node_attributes(G, score, name='score')
  return G


def betweenness_by_component(G, k, seed):
  score = {}
  count = 0
  nodes_completed = 0
  components = list(nx.connected_components(G))
  n_comp = len(components)
  n_nodes = len(G.nodes)
  for component in components:
    count += 1
    print_verbose("Working on {0} out of {1} ({2}/{3} nodes complete)".format(
      count, n_comp, nodes_completed, n_nodes))
    SG = G.__class__()
    SG.add_nodes_from((node, G.nodes[node]) for node in component)
    SG.add_edges_from((node, neighbor, data)
        for node, neighbors in G.adj.items() if node in component
        for neighbor, data in neighbors.items() if neighbor in component)
    k = k if k < len(SG.nodes) else len(SG.nodes)
    nodes_completed += len(component)
  score.update(nx.betweenness_centrality(SG, normalized=False, k=k, seed=0))
  return score



def verbose_func(verbose):
  global first_time
  if first_time:
    print("Caviar. \"The plot thickens...\" Jonathan Chang © 2021.")
    first_time = False
  def print_verbose(text):
    if verbose:
      print(text)
  return print_verbose