import os
from xml.etree import ElementTree as ET
import math
import numpy as np
from matplotlib import pyplot as plt
from svgpathtools import parse_path, Line, Arc, CubicBezier, QuadraticBezier
from tqdm import tqdm

element_types = ['path','line', 'rect', 'polyline', 'polygon', 'circle', 'ellipse']

def load_svg(path):
    """
    Parse an SVG file and return its root element, namespace map, and dimensions.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    # Extract namespace (e.g. '{http://www.w3.org/2000/svg}')
    svg_ns = root.tag.split('}')[0].strip('{')
    width = float(root.attrib.get('width', 1))
    height = float(root.attrib.get('height', 1))
    ns = {'svg': svg_ns}
    return root, ns, width, height

def parse_primitives(elem, tag, num_segments=16):
    """Break the svg elements into the fundamental primitives:
        - Line ->{'tag':'line', 'main_tag':{tag}, 'points':[(x1,y1),(x2,y2)]}
        - Arc  ->{'tag':'arc',  'main_tag':{tag}, 'center':(cx,cy), 'r':r,'start_angle':θ0,'end_angle':θ1}
        - EllipseArc ->{'tag':'ellipse_arc','main_tag':{tag}, 'center':(cx,cy),'rx':rx,'ry':ry,'start_angle':θ0,'end_angle':θ1}
        - Beziers ->{'tag':'bezier', 'main_tag':{tag}, 'points':[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
    """


    prims = []

    if tag == 'line':
        x1, y1 = float(elem.attrib.get('x1', 0)), float(elem.attrib.get('y1', 0))
        x2, y2 = float(elem.attrib.get('x2', 0)), float(elem.attrib.get('y2', 0))
        prims.append({
            'tag': 'line',
            'main_tag': tag,
            'points': [(x1, y1), (x2, y2)]
        })

    elif tag in ('polyline','polygon'):
        raw = elem.attrib.get('points','').strip()
        if raw:
            nums = raw.replace(',', ' ').split()
            coords = [(float(nums[i]), float(nums[i+1]))
                      for i in range(0, len(nums), 2)]
            end = len(coords) if tag=='polygon' else len(coords)-1
            for i in range(end):
                prims.append({
                    'tag': 'line',
                    'main_tag': tag,
                    'points': [coords[i], coords[(i+1)%end]]
                })

    elif tag == 'rect':
        x = float(elem.attrib.get('x', 0))
        y = float(elem.attrib.get('y', 0))
        w = float(elem.attrib.get('width', 0))
        h = float(elem.attrib.get('height', 0))
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        for i in range(4):
            prims.append({
                'tag': 'line',
                'main_tag': tag,
                'points': [
                    corners[i],
                    corners[(i+1) % 4]]
                })

    elif tag == 'circle':
        cx, cy, r = (float(elem.get(k, 0)) for k in ('cx','cy','r'))
        for i in range(num_segments):
            θ0 = 2*math.pi * i       / num_segments
            θ1 = 2*math.pi * (i+1)   / num_segments
            prims.append({
                'tag':         'arc',
                'main_tag':    tag,
                'center':      (cx, cy),
                'r':            r,
                'start_angle': θ0,
                'end_angle':   θ1
            })

    elif tag == 'ellipse':
        cx, cy, rx, ry = (float(elem.get(k, 0)) for k in ('cx','cy','rx','ry'))
        for i in range(num_segments):
            θ0 = 2*math.pi * i       / num_segments
            θ1 = 2*math.pi * (i+1)   / num_segments
            prims.append({
                'tag':         'ellipse_arc',
                'main_tag':    tag,
                'center':      (cx, cy),
                'rx':           rx,
                'ry':           ry,
                'start_angle': θ0,
                'end_angle':   θ1
            })

    elif tag == 'path':
        d = elem.attrib.get('d','')
        path = parse_path(d)
        for seg in path:
            if isinstance(seg, Line):
                p0 = (seg.start.real, seg.start.imag)
                p1 = (seg.end.real,   seg.end.imag)
                prims.append({
                    'tag':        'line',
                    'main_tag':    tag,
                    'points':      [p0, p1]
                })
            elif isinstance(seg, Arc):
                prims.append({
                    'tag':        'arc',
                    'main_tag':    tag,
                    'center':      (seg.center.real, seg.center.imag),
                    'r':            float(seg.radius.real),
                    'start_angle': float(seg.theta),
                    'end_angle':   float(seg.theta + seg.delta)
                })
            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                ts = np.linspace(0, 1, num_segments+1)
                bez_pts = []
                for t in ts:
                    pt = seg.point(t)
                    bez_pts.append((pt.real, pt.imag))
                prims.append({
                    'tag':    'bezier',
                    'main_tag': tag,
                    'points': bez_pts
                })

    return prims

def extract_elements(root, ns):
    """
    Parse an SVG file and extract all geometric primitives as entities.

    Each entity is a dict containing:
      - 'tag': one of ['line', 'arc', 'ellipse_arc', 'bezier']
      - primitive-specific fields (e.g. 'points', 'center', 'r', 'rx', 'ry', 'start_angle', 'end_angle')
      - 'attrib': the SVG attributes of the original element (stroke, fill, class, id, etc.)
    """
    entities = []
    for grp in root.findall('.//svg:g', ns):
        # start with the group's own attributes (e.g. fill, stroke, style)
        attrib = {k:v for k,v in grp.attrib.items() if k != 'class'}
        class_=grp.attrib.get('class','')
        for child in grp:
            tag = child.tag.split('}')[-1]
            if tag not in element_types:
                continue

            # overlay any child‐specific styles
            attrib = {k: v for k, v in child.attrib.items()
                  if k not in ('d', 'points', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'width', 'height',
                               'cx', 'cy', 'r', 'rx', 'ry')}

            # break into primitives
            prims = parse_primitives(child, tag)
            for prim in prims:
              entity = prim.copy()
              entity['attrib'] = attrib
              entity['y']=class_

              entities.append(entity)

    return entities

def normalize_entities(entities, width, height):
    """
    For each entity in-place, add:
      - for lines:   `normalized_points` = [(x/width,y/height),(x2/width,y2/height)]
      - for arcs:    `normalized_center` = (cx/width, cy/height),
                     `normalized_r`      = r / avg(width,height)
      - for ellipse_arcs:
                     `normalized_center` = (cx/width, cy/height),
                     `normalized_rx`     = rx/width,
                     `normalized_ry`     = ry/height
    """
    avg_dim = (width + height) / 2.0

    for ent in entities:
        tag = ent['tag']
        if tag == 'line' and 'points' in ent:
            # ent['points'] is [(x1,y1),(x2,y2)]
            ent['normalized_points'] = [
                (x/width, y/height) for (x,y) in ent['points']
            ]

        elif tag == 'arc':
            cx, cy = ent['center']
            r       = ent['r']
            ent['normalized_center'] = (cx/width, cy/height)
            ent['normalized_r']      = r / avg_dim

        elif tag == 'ellipse_arc':
            cx, cy = ent['center']
            rx, ry = ent['rx'], ent['ry']
            ent['normalized_center'] = (cx/width, cy/height)
            ent['normalized_rx']     = rx/width
            ent['normalized_ry']     = ry/height

        elif tag == 'bezier':
            # ent['points'] might include multiple types: start, control(s), end
            # Assume ent['points'] is a list of (x, y) tuples
            ent['normalized_points'] = [(x / width, y / height) for (x, y) in ent['points']]

        else:
            # if you introduce other tags later, handle or skip
            continue

    return entities

def start_end_connection(node1,node2):
  x1,x2=node1['normalized'][0],node1['normalized'][-1]
  y1,y2=node2['normalized'][0],node2['normalized'][-1]
  if x1==y1 or x1==y2 or x2==y1 or x2==y2:
    return True
  return False

def intersection_based_connection(node1,node2):
    """
    Checks if two line segments intersect.

    Args:
        line1: A tuple or list of two tuples/lists, each representing a point (x, y) of the first line.
        line2: A tuple or list of two tuples/lists, each representing a point (x, y) of the second line.

    Returns:
        True if the line segments intersect, False otherwise.
    """
    if start_end_connection(node1,node2):
      return 'start'
    x1, y1 = node1['normalized'][0]
    x2, y2 = node1['normalized'][1]
    x3, y3 = node2['normalized'][0]
    x4, y4 = node2['normalized'][1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return False  # Lines are parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    if (0 <= t <= 1 and 0 <= u <= 1):
      return 'intersection'
    

import networkx as nx

def connect_edges(G, method='intersection'):
    """
    Return edge index pairs (i,j) for either start_end or intersection method.
    """
    nodes = list(G.nodes())   # <-- materialize an indexable list of node IDs
    n = len(nodes)

    start_end = []
    inter    = []

    for a in range(n):
        for b in range(a+1, n):
            u = nodes[a]
            v = nodes[b]
            rel = intersection_based_connection(G.nodes[u], G.nodes[v])
            if rel == 'start':
                start_end.append((u, v))
            elif rel == 'intersection':
                inter.append((u, v))

    # Return a flattened list of edges (tuples)
    return inter + start_end if method=='intersection' else start_end  # <-- Changed this line


def build_graph(entities, method='intersection'):
    G = nx.Graph()

    # 1) add all nodes
    for idx, ent in enumerate(entities):
        if 'normalized_points' in ent:
            orient = math.atan2(ent['points'][1][1]-ent['points'][0][1],
                            ent['points'][1][0]-ent['points'][0][0])
            G.add_node(idx,
                       tag=ent['tag'],
                       main_tag=ent['main_tag'],
                       points=ent['points'],
                       attrib=ent['attrib'],
                       normalized=ent.get('normalized_points'),
                       orient=orient,
                       y=ent['y']
                       )

    # 2) compute edges
    edges = connect_edges(G,method)

    # 3) add them
    G.add_edges_from(edges)
    return G

def visualize_graph(G, use_normalized=False, ax=None, eps=int(1e2), es=1e2):
    """
    Overlay CAD shapes and graph on Matplotlib axis, but drop any segment
    whose midpoint lies too close to (0,0). Colors nodes based on predicted labels.

    Args:
      G            - NetworkX graph with node attribute 'points'
      labels       - Pandas Series of predicted labels, indexed by node_id
      method       - 'centroid' or 'first_point'
      use_normalized - unused here
      ax           - matplotlib axis (optional)
      eps          - threshold distance from origin below which we drop nodes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 25))

    # Build a dict of valid positions and node colors
    pos = {}
    node_colors = []
    for idx, data in G.nodes(data=True):
        pts = np.array(data['points'])
        mx, my = pts[0]

        # Filter out any segment whose midpoint is essentially at (0,0)
        if abs(mx) < es and abs(my) < eps:
            continue

        pos[idx] = (mx, my)

        # Get the predicted label for the node and assign a color
        label = data.get('y', -1)  # Get label, default to -1 if not found
        if label == 'Wall':
            node_colors.append('red')  # Wall
        elif label == 'Wall External':
            node_colors.append('blue') # Wall External
        elif label == 'Stairs':
            node_colors.append('purple') # stairs
        elif label == 'Window Regular':
            node_colors.append('yellow') # window regular
        elif label == 'Doors':
            node_colors.append('green') # doors
        elif label == 'Door Swing Beside':
            node_colors.append('orange') # door swing beside
        else:
            node_colors.append('gray')


    # Now only draw edges and nodes among those valid positions
    # 1) pick only edges whose both endpoints remain
    valid_edges = [(u, v) for u, v in G.edges()
                   if u in pos and v in pos]

    nx.draw_networkx_edges(G, pos, edgelist=valid_edges, edge_color='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=pos.keys(), node_size=75, node_color=node_colors, ax=ax)
    ax.invert_yaxis()
    ax.axis('off')
    return ax

def process_svg(path):
    """
    Full pipeline for single SVG: load, extract, split, normalize.
    Returns a list of entity dicts.
    """
    root, ns, width, height = load_svg(path)
    entities = extract_elements(root, ns)
    entities = normalize_entities(entities, width, height)
    G = build_graph(entities)
    ax = visualize_graph(G)
    return ax