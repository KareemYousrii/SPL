#!/usr/bin/env python

from time import time
import sys
import pickle

#import sdd

def draw_grid(model,dimension):
    for i in range(dimension):
        for j in range(dimension):
            sys.stdout.write('.')
            if j < dimension-1:
                index = i*(dimension-1) + j + 1
                sys.stdout.write('-' if model[index] else ' ')
        sys.stdout.write('\n')
        if i < dimension-1:
            for j in range(dimension):
                index = dimension*(dimension-1) + j*(dimension-1) + i + 1
                sys.stdout.write('|' if model[index] else ' ')
                sys.stdout.write(' ')
        sys.stdout.write('\n')

def print_grids(alpha,dimension,manager):
    #import pdb; pdb.set_trace()
    from inf import models
    #var_count = 2*dimension*(dimension-1)
    #var_count = 2*dimension*(dimension-1) + dimension*dimension
    print("COUNT:", sdd.sdd_model_count(alpha,manager))
    for model in models.models(alpha,sdd.sdd_manager_vtree(manager)):
        print(models.str_model(model,var_count=var_count))
        draw_grid(model,dimension)
        print("".join(str(model[x]) for x in sorted(model.keys())))

def save_grid_graph(filename,graph):
    with open(filename,'wb') as output:
        pickle.dump(graph,output)

########################################
# MAIN
########################################

# 2,14,322,23858,5735478,4468252414 ???
# 2,12,184,8512,
# 2
# 12
# 184
# 8512
# 1262816
# 575780564
# 789360053252
# 3266598486981642
# 41044208702632496804
# 1568758030464750013214100
# 182413291514248049241470885236 
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: %s [GRID-M] [GRID-N]" % sys.argv[0])
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    dimension = (dim[0]-1,dim[1]-1)
    #dimension = (1,1)

    from graphillion import GraphSet
    import graphillion.tutorial as tl
    #import networkx as nx
    #G = nx.grid_2d_graph(dim[0],dim[1])
    #G.add_edges_from([
    #    ((x, y), (x+1, y+1))
    #    for x in range(dimension[0])
    #    for y in range(dimension[1])
    #] + [
    #    ((x+1, y), (x, y+1))
    #    for x in range(dimension[0])
    #    for y in range(dimension[1])
    #])

    #import matplotlib
    #matplotlib.use("Agg")
    #import matplotlib.pyplot as plt
    #pos = dict((n, n) for n in G.nodes())
    #nx.draw_networkx(G, pos=pos)
    #plt.savefig("graph.png")

    #GraphSet.converters['to_graph'] = nx.from_edgelist
    #GraphSet.converters['to_edges'] = nx.to_edgelist
    universe = tl.grid(*dimension)
    GraphSet.set_universe(universe)

    start,goal = 1, 155
    paths = GraphSet.paths(start, goal)
    print(paths.len())

    to_exclude = [13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]
    #to_exclude = [(12, 13), (13, 26), (25, 26), (26, 39), (38, 39), (39, 52), (51, 52), (52, 65), (64, 65), (65, 78), (77, 78), (78, 91), (90, 91), (91, 104), (103, 104), (104, 117), (116, 117), (117, 130), (129, 130), (130, 143), (142, 143), (143, 156), (155, 156), (155, 168), (156, 169), (168, 169), (153, 166), (154, 167), (166, 167), (167, 168), (151, 164), (152, 165), (164, 165), (165, 166), (149, 162), (150, 163), (162, 163), (163, 164), (147, 160), (148, 161), (160, 161), (161, 162), (145, 158), (144, 157), (157, 158), (146, 159), (158, 159), (159, 160)]
    for n in to_exclude:
        paths = paths.excluding(n)
    paths = paths.smaller(33)
    
    #dim = (dimension[0]+1,dimension[1]+1)
    """ AC: SAVE ZDD TO FILE"""
    f = open("constraint.zdd" ,"w")
    paths.dump(f)
    f.close()

    with open("constraint.graph.pickle",'wb') as output:
        pickle.dump(paths.universe(), output)

    """ AC: SAVE GRAPH"""
    #nodes = [None] + [ (x,y) for x in range(dim[0]) for y in range(dim[1]) ]
    #from collections import defaultdict
    #graph = defaultdict(list)
    #for index,edge in enumerate(paths.universe()):
    #    x,y = edge
    #    x,y = nodes[x],nodes[y]
    #    graph[x].append( (index+1,y) )
    #    graph[y].append( (index+1,x) )
    #graph_filename = "%dx%d.graph.pickle" % dim
    #save_grid_graph(graph_filename,graph)

    #sdd_filename = "output/paths/paths-%d.sdd" % dimension
    #sdd_vtree_filename = "output/paths/paths-%d.vtree" % dimension
    #graph_filename = "output/paths/paths-%d.graph.pickle" % dimension
    #sdd.sdd_save(sdd_filename,alpha)
    #sdd.sdd_vtree_save(sdd_vtree_filename,sdd.sdd_manager_vtree(manager))

    #graph = _node_to_edge_map(dimension)
    #save_grid_graph(graph_filename,graph)

