#ifndef MLLIB_H
#define MLLIB_H

typedef struct {
    float *pre_activation;
    float *activation;
} layer_t;

typedef struct {
    float *weight;
    float *bias;
} parameter_t;

typedef struct {
    layer_t *layers;
    parameter_t *parameters;
} neural_network_t;

layer_t* alloc_layer (int n_layer_units);

void free_layer (layer_t *p, int n_layer_units);

parameter_t* alloc_parameter (int n_weight, int n_bias);

void free_parameter (parameter_t *p, int n_weight, int n_bias);

neural_network_t* alloc_neural_network (int n_layers, int *layer_units);

void free_neural_network (neural_network_t *p, int n_layers, int *layer_units);

#endif