from graph_tool.all import *
from graph_tool import generation as gt
import math
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib
import mpmath
import scipy as sp
import sys, os, os.path
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib  #For animation


T = 1000
b1 = 0.2 # infection probability
u1 = 0.3 # recovery probability
b2 = 0.1 # behavior change probability
u2 = 0.3 # probability of reverting behavioral change
u3 = 0.2 # media recovery rate
a = 0.8 # a*b1 is the effective infection probability for adopters of protections
p = 0 # tunes the effect of media
animation = True




seed(42)
seed_rng(42)

def deg_sampler():
    accept = False
    while not accept:
        k = np.random.randint(2, 100)
        accept = np.random.random() < k**(-2.5)
    return k
    

'''g = collection.data["netscience"]
g = GraphView(g, vfilt=label_largest_component(g), directed=False)
g = Graph(g, prune=True)

pos = g.vp["pos"]  # layout positions'''

n = 1000
g = gt.random_graph(n, deg_sampler, directed = False)
pos = sfdp_layout(g)

# The states would usually be represented with simple integers, but here we will
# use directly the color of the vertices in (R,G,B,A) format.

S = [1, 1, 1, 1]           # White color
I = [1, 0, 0, 1]           # Red color
R = [0, 0, 0, 1]           # Black color
N = [1, 1, 1, 1]           # protections not adopted, white color
A = [0, 1, 0, 1]           # protections adopted, green color

# Initialize all vertices to the S state
I_state = g.new_vertex_property("vector<double>")
for v in g.vertices():
    I_state[v] = S

#Initialize state of behavior
B_state = g.new_vertex_property("vector<double>")
for v in g.vertices():
    B_state[v] = N

I_state[randint(0, g.num_vertices()-1)] = I


# Newly infected nodes will be highlighted in red
newly_infected = g.new_vertex_property("bool")
newly_protected = g.new_vertex_property("bool")


# This creates a GTK+ window with the initial graph layout
if animation:
    win_I = GraphWindow(g, pos, geometry=(1000, 1000),
                        edge_color=[0.6, 0.6, 0.6, 1],
                        vertex_fill_color=I_state,
                        vertex_halo=newly_infected,
                        vertex_halo_color=[0, 0, 1, 0.6])
    win_B = GraphWindow(g, pos, geometry=(1000, 1000),
                        edge_color=[0.6, 0.6, 0.6, 1],
                        vertex_fill_color=B_state,
                        vertex_halo=newly_protected,
                        vertex_halo_color=[0, 0.4, 1, 1])

I_t = np.zeros(T)
t = 0
M = 0
# This function will be called repeatedly by the GTK+ main loop, and we use it
# to update the state according to the dynamics.
def update_state():
    global t
    global I_t
    global M
    newly_infected.a = False
    newly_protected.a = False

    # visit the nodes in random order
    vs = list(g.vertices())
    shuffle(vs)
    for v in vs:
        if I_state[v] == S:
            b_eff = (1 if B_state[v] == N else a)*b1
            ns = [n for n in list(v.out_neighbors()) if I_state[n] == I]
            if random() < 1 - math.exp(-min(len(ns), 5)*b_eff):
                I_state[v] = I
                B_state[v] = A
                newly_infected[v] = True
                newly_protected[v] = True
        elif I_state[v] == I:
            if random() < 1 - math.exp(-u1):
                I_state[v] = R
        
        if B_state[v] == N:
            ns = [n for n in list(v.out_neighbors()) if B_state[n] == A]
            if random() < max(1 - math.exp(-min(len(ns), 5)*b2), p*M):
                B_state[v] = A
                newly_protected[v] = True
        elif B_state[v] == A:
            if random() < 1 - math.exp(-u2):
                B_state[v] = N
                
    I_t[t] = GraphView(g, vfilt=lambda v: I_state[v] == I).num_vertices()/g.num_vertices()
    M += (1-M)*I_t[t] - u3*M
    t += 1
    


    # The following will force the re-drawing of the graph, and issue a
    # re-drawing of the GTK window.
    if animation:
        win_I.graph.regenerate_surface()
        win_I.graph.queue_draw()
        win_B.graph.regenerate_surface()
        win_B.graph.queue_draw()

    # If we are still within the time limit we need to return True
    # so that the main loop will call this function more than once.
    return t < T


# Bind the function above as an 'idle' callback.
cid = GLib.idle_add(update_state)

# We will give the user the ability to stop the program by closing the window.
if animation:
    win_I.connect("delete_event", Gtk.main_quit)
    win_B.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
if animation:
    win_I.show_all()
    win_B.show_all()
Gtk.main()
#plt.plot(range(T), I_t)
#plt.show()