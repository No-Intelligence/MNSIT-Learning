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

typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
} activation_t;

typedef struct {
    float *delta;
} layre_grad_t;

typedef struct {
    float *weight_grad;
    float *bias_grad;
} param_grad_t;

typedef struct {
    layer_t *layers;
    parameter_t *parameters;
    activation_t *activations;
    layre_grad_t *layer_grad;
    param_grad_t *param_grad;
    int n_layers;
    int *layer_size;
} neural_network_t;

layer_t* alloc_layer (int n_layers, int *layer_size);

void free_layer (layer_t *layer, int n_layers, int *layer_size);

parameter_t* alloc_parameter (int n_layers, int *layer_size);

void free_parameter (parameter_t *parameter, int n_layers, int *layer_size);

layre_grad_t* alloc_layer_grad (int n_layers, int *layer_size);

void free_layer_grad (layre_grad_t *layer_grad, int n_layers, int *layer_size);

param_grad_t* alloc_param_grad (int n_layers, int *layer_size);

void free_param_grad (param_grad_t *param_grad, int n_layers, int *layer_size);

/**
 * 全結合のニューラルネットワークを作成します。
 * @param n_layers 入力層・出力層両方を含めた層の数。(int)
 * @param layer_size 各層のサイズ。(int配列)
 * @param activations 使用する活性化関数、層の数より一つ少ない(activation配列)
 */
neural_network_t* alloc_neural_network (int n_layers, int *layer_size, activation_t *activations);

void free_neural_network (neural_network_t *neural_network);

void matrix_arr_mul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr);

void add_array (float *operated_arr, float *input_arr, int n_of_arr);

float extract_max (float *input_array, int n_of_input_arr);

void relu (float *input_arr, float *output_arr, int n_of_arr);

void softmax (float *input_arr, float *output_arr, int n_of_arr);

void he_initialize (float *weight, int fan_in, int fan_out);

void compute_output_delta (float *output, float *output_layer, float *answer_arr, int n_of_arr);

void compute_hidden_delta (float *z_delta, float *next_weight, float *z, float *output, int n_of_activation, int n_of_z_delta);

void compute_weight_grad (float *z_delta, float *previous_activation_arr, float *output_arr, int n_of_output, int n_of_input);

void compute_bias_grad (float *input, float *output, int n_of_arr);

void forward_pass (neural_network_t *neural_network, float *input, float *output);

void parameter_initialize (neural_network_t *neural_network);

void backward_pass (neural_network_t *neural_network, float *answer);

void updata_param (neural_network_t *neural_network, float *learning_rate);


#endif