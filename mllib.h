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
    layer_t *layers;
    parameter_t *parameters;
    activation_t *activations;
    int n_layers;
    int *layer_size;
} neural_network_t;

layer_t* alloc_layer (int n_layers, int *layer_size);

void free_layer (layer_t *layer, int n_layers, int *layer_size);

parameter_t* alloc_parameter (int n_layers, int *layer_size);

void free_parameter (parameter_t *parameter, int n_layers, int *layer_size);

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

void forward_pass (neural_network_t *neural_network, float *input, float *ouput);

#endif