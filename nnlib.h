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
    int output_size;
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

void matrix_arr_mul (float *restrict output_arr, const float *restrict input_arr, const float *restrict matrix, int n_of_output_arr, int n_of_input_arr);

void add_array (float *restrict operated_arr, const float *restrict input_arr, int n_of_arr);

void relu (const float *restrict input_arr, float *restrict output_arr, int n_of_arr);

void leaky_relu (const float *restrict input_arr, float *restrict output_arr, int n_of_arr);

float extract_max (const float *restrict input_array, int n_of_input_arr);

void softmax (const float *restrict input_arr, float *restrict output_arr, int n_of_arr);

void forward_convolution (const float *restrict input, const float *restrict filter, float *restrict output, int n_input_height, int n_input_width, int n_input_channel, int filter_height, int filter_width, int n_filters, int stride, const float *restrict bias);

void forward_maxpool(const float *restrict input, float *restrict output, int n_channels, int in_height, int in_width, int kernel_height, int kernel_width, uint8_t *restrict mask);

void forward_pass (neural_network_t *nn, const float *restrict input);

void compute_output_softmax_delta (float *restrict output_delta, const float *restrict output_layer_activation, const float *restrict answer_arr, int n_of_arr);

void compute_backward_fc (float *restrict output_delta, const float *restrict current_delta, const float *restrict weight, int n_output_delta, int n_current_delta);

void compute_weight_grad (const float *restrict z_delta, const float *restrict previous_activation_arr, float *restrict output_arr, int n_of_output, int n_of_input);

void compute_bias_grad (float *restrict output_bias_grad, const float *restrict delta, int n_of_arr);

void compute_backward_maxpool (float *restrict computed_delta, const float *restrict current_delta, const uint8_t *restrict mask, int n_channels, int in_h, int in_w);

void compute_backward_conv (float *restrict computed_delta, float *restrict grad_filter, float *restrict grad_bias, const float *restrict activation, const float *restrict current_delta, const float *restrict filter, const float *restrict input, int n_input_height, int n_input_width, int filter_height, int filter_width, int n_filters, int in_channel, int in_h, int in_w, int stride);

void backward_pass (neural_network_t *nn, const float *restrict input, const float *restrict answer);

void parameter_initialize (neural_network_t *nn);

void update_param_adam (neural_network_t *nn, float lr, float weight_decay, float beta1, float beta2, float eps, int t, int batch_size);

#endif
