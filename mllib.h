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

/**
 * ネットワークを開放・破棄します。
 */
void free_neural_network (neural_network_t *neural_network);

void matrix_arr_mul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr);

void add_array (float *operated_arr, float *input_arr, int n_of_arr);

float extract_max (float *input_array, int n_of_input_arr);

void relu (float *input_arr, float *output_arr, int n_of_arr);

void softmax (float *input_arr, float *output_arr, int n_of_arr);

void he_initialize (float *weight, int fan_in, int fan_out);

void compute_output_delta (float *output_delta, float *output_layer_activation, float *answer_arr, int n_of_arr);

void compute_hidden_delta (float *output_delta, float *current_delta, float *current_weight, float *backward_pre_activation, int n_of_activation, int n_of_z_delta);

void compute_weight_grad (float *z_delta, float *previous_activation_arr, float *output_arr, int n_of_output, int n_of_input);

void compute_bias_grad (float *input, float *output, int n_of_arr);

/**
 * ニューラルネットワークの順伝播を計算します。
 * @param input 入力となるfloat型の配列です。例えば、MNISTデータセットなら784個の数値になります。
 * @param output 出力を格納するfloat型の配列です。例えば、MNISTデータセットなら10個の数値になります。
 */
void forward_pass (neural_network_t *neural_network, float *input, float *output);

/**
 * ニューラルネットワークの学習前に必要な各パラメータの初期化を行います。
 */
void parameter_initialize (neural_network_t *neural_network);

/**
 * ニューラルネットワークの逆伝播を計算し、勾配を求めます。
 * @param answer 正解となる配列です。例えば、MNISTデータセットなら正解が1、それ以外が0の配列になります。
 */
void backward_pass (neural_network_t *neural_network, float *answer);

/**
 * ニューラルネットワークのパラメータを更新します。
 * @param learning_rate モデルの学習率です。高いほど更新幅が大きくなりますが、大きすぎると学習が不安定になる・過学習が起きる・学習が発散するなどの副作用があります。
 * @param regularization_rate モデルのL2正規化の強さを決める数値です。0.0でオフにできます。高いと過学習が抑えられ新規のデータに対応しやすくなりますが、大きすぎると未学習になることがあります。
 */
void update_param (neural_network_t *neural_network, float learning_rate, float regularization_rate);

/**
 * ニューラルネットワークのパラメータを保存します。成功で0、失敗で-1を返します。
 */
int save_neural_network(const neural_network_t *neural_network, const char *filename);
/**
 * ニューラルネットワークのパラメータを読み込みます。成功で0、失敗で-1を返します。
 */
int load_neural_network(neural_network_t *neural_network, const char *filename);

#endif