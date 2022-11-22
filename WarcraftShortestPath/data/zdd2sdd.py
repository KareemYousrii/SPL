#!/usr/bin/env python

from vtrees import vtrees
import sdd
import time

def global_model_count(alpha,manager):
    mc = sdd.sdd_model_count(alpha,manager)
    var_count = sdd.sdd_manager_var_count(manager)
    var_used = sdd.sdd_variables(alpha,manager)
    for used in var_used[1:]:
        if not used:
            mc = 2*mc
    return mc

def zero_normalize_sdd(alpha,alpha_vtree,vtree,manager):
    if sdd.sdd_node_is_false(alpha): return alpha

    #if vtree == sdd.sdd_vtree_of(alpha):
    if vtree == alpha_vtree:
        return alpha

    if sdd.sdd_vtree_is_leaf(vtree):
        var = sdd.sdd_vtree_var(vtree)
        nlit = sdd.sdd_manager_literal(-var,manager)
        return nlit

    left,right = sdd.sdd_vtree_left(vtree),sdd.sdd_vtree_right(vtree)
    beta_left  = zero_normalize_sdd(alpha,alpha_vtree,left,manager)
    beta_right = zero_normalize_sdd(alpha,alpha_vtree,right,manager)
    beta = sdd.sdd_conjoin(beta_left,beta_right,manager)
    return beta

def pre_parse_bdd(filename):
    f = open(filename)
    node_count = 0
    bdd_vars = set()
    for line in f.readlines():
        if line.startswith("."): break
        node_count += 1
        line = line.strip().split()
        var = int(line[1])
        bdd_vars.add(var)
    f.close()

    #return len(bdd_vars),node_count
    return max(bdd_vars),node_count

def parse_bdd(filename):

    var_count,node_count = pre_parse_bdd(filename)
    print "   zdd var count:", var_count
    print "  zdd node count:", node_count

    manager = start_manager(var_count,range(1,var_count+1))
    root = sdd.sdd_manager_vtree(manager)
    nodes = [None] * (node_count+1)
    index,id2index = 1,{}

    f = open(filename)
    for line in f.readlines():
        if line.startswith("."): break
        line = line.strip().split()
        nid = int(line[0])
        dvar = int(line[1])
        lo,hi = line[2],line[3]

        hi_lit = sdd.sdd_manager_literal( dvar,manager)
        lo_lit = sdd.sdd_manager_literal(-dvar,manager)

        if   lo == 'T':
            lo_sdd,lo_vtree = sdd.sdd_manager_true(manager),None
        elif lo == 'B':
            lo_sdd,lo_vtree = sdd.sdd_manager_false(manager),None
        else:
            lo_id = int(lo)
            lo_sdd,lo_vtree = nodes[id2index[lo_id]]

        if   hi == 'T':
            hi_sdd,hi_vtree = sdd.sdd_manager_true(manager),None
        elif hi == 'B':
            hi_sdd,hi_vtree = sdd.sdd_manager_false(manager),None
        else:
            hi_id = int(hi)
            hi_sdd,hi_vtree = nodes[id2index[hi_id]]

        #v1,v2 = sdd.sdd_vtree_of(hi_lit),sdd.sdd_vtree_of(hi_sdd)
        #vt = sdd.sdd_vtree_lca(v1,v2,root)

        if var_count > 1:
            vt = sdd.sdd_manager_vtree_of_var(dvar,manager)
            vt = sdd.sdd_vtree_parent(vt)
            vt = sdd.sdd_vtree_right(vt)
        else:
            vt = None

        if dvar < var_count:
            hi_sdd = zero_normalize_sdd(hi_sdd,hi_vtree,vt,manager)
            lo_sdd = zero_normalize_sdd(lo_sdd,lo_vtree,vt,manager)
            vt = sdd.sdd_vtree_parent(vt)

        hi_sdd = sdd.sdd_conjoin(hi_lit,hi_sdd,manager)
        lo_sdd = sdd.sdd_conjoin(lo_lit,lo_sdd,manager)
        alpha = sdd.sdd_disjoin(hi_sdd,lo_sdd,manager)

        nodes[index] = (alpha,vt)
        id2index[nid] = index
        index += 1
            
    f.close()

    return manager,nodes[-1][0]

def start_manager(var_count,order):
    #vtree = sdd.sdd_vtree_new_with_var_order(var_count,order,"right")
    vtree = sdd.sdd_vtree_new(var_count,"right")
    #vtree = vtrees.right_linear_vtree(1,var_count+1)
    manager = sdd.sdd_manager_new(vtree)
    sdd.sdd_manager_auto_gc_and_minimize_off(manager)
    sdd.sdd_vtree_free(vtree)
    return manager

########################################
# MAIN
########################################

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print "usage: %s [BDD_FILENAME]" % sys.argv[0]
        exit(1)

    filename = sys.argv[1]

    print "=== reading", filename

    start = time.time()
    manager,alpha = parse_bdd(filename)
    end = time.time()
    print "      sdd node count: %d" % sdd.sdd_count(alpha)
    print "            sdd size: %d" % sdd.sdd_size(alpha)
    print "     sdd model count: %d" % sdd.sdd_model_count(alpha,manager)
    print "  global model count: %d" % global_model_count(alpha,manager)
    print "       read bdd time: %.3fs" % (end-start)

    sdd.sdd_ref(alpha,manager)
    start = time.time()
    sdd.sdd_manager_minimize(manager)
    end = time.time()
    print "  min sdd node count: %d" % sdd.sdd_count(alpha)
    print "        min sdd size: %d" % sdd.sdd_size(alpha)
    print "        min sdd time: %.3fs" % (end-start)
    sdd.sdd_deref(alpha,manager)

    sdd.sdd_save(filename + ".sdd",alpha)
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    vtree = sdd.sdd_manager_vtree(manager)
    sdd.sdd_vtree_save(filename + ".vtree",vtree)
    #sdd.sdd_vtree_save_as_dot(filename +".vtree.dot",vtree)

    """
    print "===================="
    print "before garbage collecting..." 
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)
    print "garbage collecting..."
    sdd.sdd_manager_garbage_collect(manager)
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)
    """
