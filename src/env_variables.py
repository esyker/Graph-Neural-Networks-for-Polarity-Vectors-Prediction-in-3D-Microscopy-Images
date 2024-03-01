###################################################
### Define int encodings for string variables #####
###################################################
global nodes_type_int_encodings
global reversed_nodes_type_int_encodings
#nodes type encodings
nodes_type_int_encodings = {
    "nuclei":0,
    "golgi":1
}

reversed_nodes_type_int_encodings = {v: k for k, v in nodes_type_int_encodings.items()}

global edges_type_int_encodings
global reversed_edges_type_int_encodings
#edges type encodings
edges_type_int_encodings = {
    "nuclei-nuclei":0,
    "golgi-golgi":1,
    "golgi-nuclei":2,
    "nuclei-golgi":3
}
reversed_edges_type_int_encodings = {v: k for k, v in edges_type_int_encodings.items()}