#include "nbody.h"

#include <mpi.h>

#include <stdlib.h>

#include <stdio.h>

int get_object_count(Node** nodes, int node_count) {
    int object_count = 0;
    for(int i=0; i<node_count; i++) {
        object_count += nodes[i]->leaf.count;
    }
    return object_count;
}

void copy_high_res_data(Node* node, float high_res_data[5*node->leaf.count+1]) {
    for(int i = 0; i < node->leaf.count; i++) {
        high_res_data[5*i] = node->leaf.x[i];
        high_res_data[5*i+1] = node->leaf.y[i];
        high_res_data[5*i+2] = node->leaf.vx[i];
        high_res_data[5*i+3] = node->leaf.vy[i];
        high_res_data[5*i+4] = node->leaf.mass[i];
    }
    high_res_data[5*node->leaf.count] = node->leaf.timestamp_high_res;
}

void copy_high_res_data_into_node(Node* node, float* high_res_data, 
    int obj_count) {
    
    node_leaf_ensure_capacity(node, obj_count);

    node->leaf.count = obj_count;

    for(int i = 0; i < obj_count; i++) {
        node->leaf.x[i]     = high_res_data[5*i];
        node->leaf.y[i]     = high_res_data[5*i+1];
        node->leaf.vx[i]    = high_res_data[5*i+2];
        node->leaf.vy[i]    = high_res_data[5*i+3];
        node->leaf.mass[i]  = high_res_data[5*i+4];
    }
    node->leaf.timestamp_high_res = high_res_data[5*obj_count];
}


void simulate_par(int steps, int num_leaves, Node* root) {
    // root is the root node of the whole domain. The data will be initialised in each process, so
    // you do not need to copy it at the start. However, the main process (rank 0) has to have the
    // full data at the end of the simulation, so you will need to collect it.

    int rank, num_procs;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;

    Node** nodes = malloc(sizeof(Node*) * num_leaves);
    int node_count = 0;
    node_find_nearby(nodes, &node_count, root, root);

    int object_count = get_object_count(nodes,node_count);

    Node* node = nodes[rank];

    Node** neighbors = malloc(sizeof(Node*) * num_leaves);
    int neighbor_count = 0;
    node_find_nearby(neighbors, &neighbor_count, node, root);

 
    float leaving[5*MAX_LEAVING];
    float** leavingTo;
    int* leavingToNum;
    leavingTo = malloc(sizeof(float*) * node_count);
    leavingToNum = malloc(sizeof(float) * node_count);
    for(int i=0; i<node_count; i++) {
        leavingTo[i] = malloc(sizeof(float) * 5 * MAX_LEAVING);
    }
    
    MPI_Request reqs[31];

    for (int step = 0; step < steps; ++step) {

        for(int i=0; i<node_count; i++)
            leavingToNum[i] = 0;
        
        
        compute_acceleration(node, root);
        
        int left = move_objects(node, leaving);
        

        for (int i = 0; i < left; ++i) {
            for (int n = 0; n < node_count; ++n) {
                if (node_contains_point(nodes[n], leaving[5*i], 
                        leaving[5*i+1])) {

                    leavingTo[n][5*leavingToNum[n]]   = leaving[5*i];
                    leavingTo[n][5*leavingToNum[n]+1] = leaving[5*i+1];
                    leavingTo[n][5*leavingToNum[n]+2] = leaving[5*i+2];
                    leavingTo[n][5*leavingToNum[n]+3] = leaving[5*i+3];
                    leavingTo[n][5*leavingToNum[n]+4] = leaving[5*i+4]; 
                    leavingToNum[n]++;

                    break;
                }
            }
        }

        // Send leaving object info to other nodes
        for(int i=0; i<node_count; i++) {
            if(i != rank) {
                MPI_Isend(leavingTo[i], 5 * leavingToNum[i], MPI_FLOAT, 
                    nodes[i]->id, 0, MPI_COMM_WORLD, &reqs[i]);
            }
        }

        float entering[5*MAX_LEAVING];
        // Receive entering object info from other nodes
        for(int i=0; i<node_count; i++) {
            if(rank == i) continue;

            MPI_Recv(entering, 5*MAX_LEAVING, MPI_FLOAT, nodes[i]->id, 0, 
                MPI_COMM_WORLD, &stat);
        
            int entering_count = 0;
            MPI_Get_count(&stat, MPI_FLOAT, &entering_count);
            entering_count /= 5;
            for(int j=0; j<entering_count; j++) {
                node_leaf_append_object(node, entering[5*j], entering[5*j+1], 
                    entering[5*j+2], entering[5*j+3], entering[5*j+4]);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
 
        float high_res_data[5 * node->leaf.count + 1];
        copy_high_res_data(node, high_res_data);
        // Share high res data of this node with the neighbors
        for(int i=0; i<neighbor_count; i++) {
            MPI_Isend(high_res_data, 5*node->leaf.count+1, MPI_FLOAT, 
                neighbors[i]->id, 0, MPI_COMM_WORLD, &reqs[i]);
        }

        // Receive high res data of the neighbors
        for(int i=0; i<neighbor_count; i++) {
            float high_res_data_from_neighbor[5 * object_count + 1];
            MPI_Recv(high_res_data_from_neighbor, 5 * object_count + 1, MPI_FLOAT, 
                neighbors[i]->id, 0, MPI_COMM_WORLD, &stat);
            
            int neighbor_obj_count = 0;
            MPI_Get_count(&stat, MPI_FLOAT, &neighbor_obj_count);
            neighbor_obj_count--;
            neighbor_obj_count /= 5;
            copy_high_res_data_into_node(neighbors[i], 
                high_res_data_from_neighbor, neighbor_obj_count);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        node_update_low_res(node);

        // Share low res data of this node with every other node
        float low_res_data[4]; // 4:x,y,mass,timestamp
        low_res_data[0] = node->low_res_x;
        low_res_data[1] = node->low_res_y;
        low_res_data[2] = node->low_res_mass;
        low_res_data[3] = node->timestamp_low_res;
        
        
        for(int i=0; i<31; i++) {
            if(i != rank)
                MPI_Isend(low_res_data, 4, MPI_FLOAT, i, 0, MPI_COMM_WORLD, 
                    &reqs[i]);
        }

        // Receive low res data of every other node
        for(int i=0; i<31; i++) {
            if(i != rank) {
                MPI_Recv(low_res_data, 4, MPI_FLOAT, i, 0, MPI_COMM_WORLD, 
                    MPI_STATUS_IGNORE);
                nodes[i]->low_res_x = low_res_data[0];
                nodes[i]->low_res_y = low_res_data[1];
                nodes[i]->low_res_mass = low_res_data[2];
                nodes[i]->timestamp_low_res = low_res_data[3];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        node_update_low_res(root);
    } 

    
    free(leavingToNum);
    free(leavingTo);
    free(nodes);
}
