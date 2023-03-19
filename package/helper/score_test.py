import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
from scipy.stats import pearsonr
import math
import copy
import itertools

def add_weight(G, range_wt=(0.01, 1)):
    for (u,v) in G.edges():
        G[u][v]['weight'] = round(random.uniform(range_wt[0], range_wt[1]), 
                                       2)

    return G



def add_weight_range(G_list, range_wt=(0.01, 1)):
    all_G = []
    
    for G in G_list:
        for i in np.arange(range_wt[0], range_wt[1], 0.01):
            G_new = copy.deepcopy(G)

            for (u,v) in G_new.edges():
                G_new[u][v]['weight'] = i

            all_G.append(G_new)
        
    return all_G


    
def line_graph(length, 
               weights=True,
               range_wt=(0.01,1)
              ):
    print('\n ---------- Start: Line graph --------- \n')
    
    G_path = nx.path_graph(length)

    # nx.draw(G_path)
    
    # if weights == True:
    #     return add_weight(G_path, range_wt)
    # else:
    #     return add_weight(G_path, (1,1))

    return G_path
    
def star_graph(size, 
               weights=True,
               range_wt=(0.01,1)
              ):
    G_star = nx.star_graph(size)
    
    # nx.draw(G_star)
    
    # print('\n Star graph')
    
    # if weights == True:
    #     return add_weight(G_star, range_wt)
    # else:
    #     return add_weight(G_star, (1,1))
    
    return G_star
    
    
def complete_graph(size,
                   weights=True,
                   range_wt=(0.01,1)
                  ):
    G_comp = nx.complete_graph(size)
    
    # nx.draw(G_comp)
    
    # print('\n Complete graph')
    
    # if weights == True:
    #     return add_weight(G_comp, range_wt)
    # else:
    #     return add_weight(G_comp, (1,1))
    
    return G_comp


def remove_edge(G):
    edges = list(G.edges)
    # nonedges = list(nx.non_edges(G))
    chosen_edge = random.choice(edges)
    G.remove_edge(chosen_edge[0], chosen_edge[1])
    
    return G


def sum_weights(G):
    sum_wt = 0
    for (u,v,w) in G.edges(data=True):
        sum_wt = sum_wt + float(w['weight'])
        
    return round(sum_wt, 2)




def create_different_graph_version(G, loop):
    G_new = copy.deepcopy(G)
    
    removed_list = []
    for l in range(loop):
        G_new = remove_edge(G_new)
        
        if nx.is_empty(G_new):
            break
        else:
            G_new.remove_nodes_from(list(nx.isolates(G_new)))
        
        removed_list.append(copy.deepcopy(G_new))
        

    return removed_list
   
            
def include_graphs(num_node):
    loop = num_node * (num_node - 1)
    print('\n ---------- Start: Line graph --------- \n')
    
    # G_l = line_graph(length = num_node)
    # G_l_list = create_different_graph_version(G_l, loop)
    # G_l_wt_list = add_weight_range(G_l_list)
    
    print('\n ---------- End: Line graph   --------- \n')
    
    print('\n ---------- Start: Star graph --------- \n')
    
    # G_s = star_graph(size = num_node)
    # G_s_list = create_different_graph_version(G_s, loop)
    # G_s_wt_list = add_weight_range(G_s_list)
    
    print('\n ---------- End: Star graph   --------- \n')
    
    print('\n ---------- Start: Complete graph --------- \n')
    
    G_c = complete_graph(size = num_node)
    G_c_list = create_different_graph_version(G_c, loop)
    G_c_wt_list = add_weight_range(G_c_list)
    
    print('\n ---------- End: Complete graph   --------- \n')
    
    return [G_c_wt_list]
    

    
def avg_weight(G_set):
    sum_wt = 0
    n = 0
    
    for G in G_set:
        n = n + G.number_of_nodes()
        
        for (u,v,w) in G.edges(data=True):
            sum_wt = sum_wt + float(w['weight'])
        
    return [round(sum_wt, 2)/(n), n]

#clustering is calculated for each component individually
def average_clustering(G_set):
    sum_avg = 0
    for G in G_set:
        sum_avg = sum_avg + nx.average_clustering(G)
        
    return sum_avg

#all components are considered one unit and clustering coefficients are calcualted for overall components
def total_clustering(G_set):
    
    for i, G in enumerate(G_set):
        if i == 0:
            G_union = G_set[0]
            continue
            
        G_union = nx.disjoint_union(G, G_union)
    
    return nx.average_clustering(G_union)

def coordination_score(G_sets, display=True):
    if display:
        print('Number of combinations :', len(G_sets))
       
    for G_set in G_sets:
        avg_wt, n =  avg_weight(G_set)
        avg_clust = average_clustering(G_set)
            
        coord_score = avg_wt + math.log10(n) + avg_clust
    
#     if display:
#         print('\n Coordination score :', coord_score)
        
#     return coord_score

def transitivity_vs_clustering(G_set, output_path):
    trans = []
    clust = []
    n = []
    
    for G in G_set:
        trans.append(nx.transitivity(G))
        clust.append(nx.average_clustering(G))
        n.append(G.number_of_nodes())
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(trans, n, c='b', label='Transitivity')
    ax.scatter(clust, n, c='r', label='Avg. Clustering')
    
    ax.set_xlabel('Clustering coefficient')
    ax.set_ylabel('Number of nodes')
    
    plt.legend(loc='upper left');
    plt.show()
               
    fig.savefig(f'{output_path}/transitivity_vs_avg_clustering.png', 
      facecolor='white', 
      transparent=False)
    
    
def combine_all(graph_list):
    two_comb = list(itertools.product(*graph_list))
    print('three combinatins ', two_comb[0])
    
    return two_comb
    
    
def generate_data(num_node, output_path):
    # data = [line_graph(node_range),
    
    #Line graph, complete graph
    graph_list = include_graphs(num_node)
    
    # print('here', len(graph_list[0]), len(graph_list[1]))
    
    transitivity_vs_clustering(graph_list[0], output_path)
    # two_comb = combine_all(graph_list)
    
    # coordination_score(two_comb)
    
    return graph_list
# def calculate_score(data, output_path):