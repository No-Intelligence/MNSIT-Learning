#include "mllib.h"
#include <stdlib.h>
#include <string.h>

layer_t* alloc_layer (int n_layers, int *layer_size) {
    layer_t *p = calloc(n_layers, sizeof(layer_t));
    for (size_t i = 0; i < n_layers; i++)
    {
        p[i].pre_activation = calloc(layer_size[i], sizeof(float));
        p[i].activation = calloc(layer_size[i], sizeof(float));
    }
    return p;
}

void free_layer (layer_t *layer, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers; i++)
    {
        free(layer[i].pre_activation);
        free(layer[i].activation);
    }
    free(layer);
}

parameter_t* alloc_parameter (int n_layers, int *layer_size) {
    parameter_t *p = calloc(n_layers - 1, sizeof(parameter_t));
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        p[i].weight = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].bias = calloc(layer_size[i + 1], sizeof(float));
    }
    return p;
}

void free_parameter (parameter_t *parameter, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        free(parameter[i].weight);
        free(parameter[i].bias);
    }
    free(parameter);
}

neural_network_t* alloc_neural_network (int n_layers, int *layer_size, activation_t *activations) {
    neural_network_t *p = calloc(1, sizeof(neural_network_t));
    p->layers = alloc_layer(n_layers, layer_size);
    p->parameters = alloc_parameter(n_layers, layer_size);
    p->activations = calloc(n_layers - 1, sizeof(activation_t));
    memcpy(p->activations, activations, (n_layers - 1) * sizeof(activation_t));
    p->n_layers = n_layers;
    p->layer_size = calloc(n_layers, sizeof(int));
    memcpy(p->layer_size, layer_size, n_layers * sizeof(int));
    return p;
}

void free_neural_network (neural_network_t *neural_network) {
    free_layer(neural_network->layers, neural_network->n_layers, neural_network->layer_size);
    free_parameter(neural_network->parameters, neural_network->n_layers, neural_network->layer_size);
    free(neural_network->activations);
    free(neural_network->layer_size);
    
    free(neural_network);
}