#ifndef IMAGE_PROCESS_NEURAL_NETWORK_LIBRARY_H
#define IMAGE_PROCESS_NEURAL_NETWORK_LIBRARY_H

#include <stdint.h>

typedef enum {
    LAYER_FC,
    LAYER_CONV,
    LAYER_POOL,
    LAYER_RELU,
    LAYER_LEAKY_RELU,
    LAYER_SOFTMAX,
    LAYER_FLATTEN,
} layer_type_t;

typedef struct {
    int in_size;
    int out_size;
    float *weight;
    float *bias;
    float *m_weight;
    float *v_weight;
    float *m_bias;
    float *v_bias;
    float *grad_weight;
    float *total_grad_weight;
    float *grad_bias;
    float *total_grad_bias;
} fc_layer_t;

typedef struct {
    int in_height;
    int in_width;
    int in_channel;
    int filter_height;
    int filter_width;
    int n_filters;
    int filter_stride;
    int n_padding;
    float *filter;
    float *m_filter;
    float *v_filter;
    float *bias;
    float *m_bias;
    float *v_bias;
    float *grad_filter;
    float *total_grad_filter;
    float *grad_bias;
    float *total_grad_bias;
} conv_layer_t;

typedef struct {
    int in_height;
    int in_width;
    int in_channel;
    int kernel_height;
    int kernel_width;
    uint8_t *mask;
} pool_layer_t;

typedef struct {
    layer_type_t type;
    float output_size;
    union {
        fc_layer_t fc;
        conv_layer_t conv;
        pool_layer_t pool;
    } data;
    float *output;
    float *delta;
} layer_t;

typedef struct {
    layer_t *layers;
    int n_layers;
} neural_network_t;

neural_network_t* alloc_neural_network (void);

void add_fc_layer (neural_network_t *nn, int in_size, int out_size);

void add_conv_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int filter_height, int filter_width, int n_filters, int filter_stride, int n_padding);

void add_pool_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int kernel_height, int kernel_width);

void add_activation_layer (neural_network_t *nn, layer_type_t activation);

void add_flatten_layer (neural_network_t *nn);

void free_neural_network (neural_network_t *nn);

void matrix_arr_mul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr);

void add_array (float *operated_arr, float *input_arr, int n_of_arr);

void relu (float *input_arr, float *output_arr, int n_of_arr);

void leaky_relu (float *input_arr, float *output_arr, int n_of_arr);

float extract_max (float *input_array, int n_of_input_arr);

void softmax (float *input_arr, float *output_arr, int n_of_arr);

void forward_convolution (float *input, float *filter, float *output, int n_input_height, int n_input_width, int n_input_channel, int filter_height, int filter_width, int n_filters, int stride, float *bias);

void forward_maxpool(float *input, float *output, int n_channels, int in_height, int in_width, int kernel_height, int kernel_width, uint8_t *mask);

void forward_pass (neural_network_t *nn, float *input);

#endif