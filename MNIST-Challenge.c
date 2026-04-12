#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

// ブロックサイズ（L1/L2キャッシュサイズに応じて調整）
#define BLOCK_H 64
#define BLOCK_W 64

#define PI 3.14159265358979
#define n_of_input_layer 1600
#define n_of_first_hidden_layer 128
#define n_of_output_layer 10
#define learning_rate 0.001
#define momentum_beta 0.9f
#define batch_size 300
#define epoch 1
#define debug 1
#define neck_check 0
#define threaded 1
#define num_threads 6
#define regularization_rate 0.0005f
#define dropout 0
#define dropout_rate 0.3f
#define avx2 false

//convolutional layer param
#define filter_hight 3
#define filter_width 3
#define n_of_first_channel 32
#define n_of_second_channel 64

#define train_images "train-images.idx3-ubyte"
#define train_labels "train-labels.idx1-ubyte"
#define test_images "t10k-images.idx3-ubyte"
#define test_labels "t10k-labels.idx1-ubyte"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    float *layer;
    int *mask;
} maxpool_layer_t;

typedef struct {
    float *filter;
} conv_filter_t;

typedef struct {
    float *layer;
} conv_layer_t;

typedef struct {
    //buffer
    uint8_t *training_image_buffer, *training_label_buffer;

    //weights
    conv_filter_t *first_conv_filter, *second_conv_filter;
    float *first_conv_bias, *second_conv_bias;
    float  *w1, *wout, *b1, *bout;

    //pre-activations
    conv_layer_t *first_conv_layer_pre_activation, *second_conv_layer_pre_activation;
    float *z_1, *z_out;

    //activations
    conv_layer_t *first_conv_layer_activation, *second_conv_layer_activation;
    maxpool_layer_t *first_maxpooling_layer, *second_maxpooling_layer;
    float *a_in, *a_0, *a_1, *a_out;

    //deltas
    float *delta_1, *delta_4, *delta_in;
    maxpool_layer_t *backward_second_maxpool, *backward_first_maxpool;
    conv_layer_t *backward_second_conv, *backward_first_conv;

    //gradient
    float *grad_w1, *grad_w4, *grad_b1, *grad_b4, *grad_to_b_conv1, *grad_to_b_conv2;
    conv_filter_t *grad_to_second_conv_filter, *grad_to_first_conv_filter;

    //gradient total
    float *grad_w1t, *grad_w4t, *grad_b1t, *grad_b4t, *grad_to_b_conv1_t, *grad_to_b_conv2_t;
    conv_filter_t *grad_to_first_conv_filter_t, *grad_to_second_conv_filter_t;

    //return buffer
    float *return_grad_w1t, *return_grad_w4t, *return_grad_b1t, *return_grad_b4t, *return_grad_to_b_conv1_t, *return_grad_to_b_conv2_t;
    conv_filter_t *return_grad_to_first_conv_filter_t, *return_grad_to_second_conv_filter_t;

} thread_workspace_t;

maxpool_layer_t* alloc_maxpool_layer (int n_channels, int h, int w) {
    maxpool_layer_t *p = calloc(n_channels, sizeof(maxpool_layer_t));
    for (size_t i = 0; i < n_channels; i++)
    {
        p[i].layer = malloc(h * w * sizeof(float));
        p[i].mask = malloc(h * w * sizeof(int));
    }
    return p;
}

void free_maxpool_layer (maxpool_layer_t *p, int n_channels) {
    for (size_t i = 0; i < n_channels; i++)
    {
        free(p[i].layer);
        free(p[i].mask);
    }
    free(p);
}

conv_filter_t* alloc_filter (int n_filters) {
    conv_filter_t *tmp_filter = calloc(n_filters, sizeof(conv_filter_t));
    for (size_t i = 0; i < n_filters; i++)
    {
        tmp_filter[i].filter = calloc(filter_hight * filter_width, sizeof(float));
    }
    return tmp_filter;
}

void free_filter (conv_filter_t *tmp_filter, int n_filters) {
    for (size_t i = 0; i < n_filters; i++)
    {
        free(tmp_filter[i].filter);
    }
    free(tmp_filter);
}

conv_layer_t* alloc_conv_layer (int channels, int image_hight, int image_width) {
    conv_layer_t *tmp_layer = calloc(channels, sizeof(conv_layer_t));
    for (size_t i = 0; i < channels; i++)
    {
        tmp_layer[i].layer = malloc(image_hight * image_width * sizeof(float));
    }
    return tmp_layer;
}

void free_conv_layer (conv_layer_t *tmp_layer, int n_images) {
    for (size_t i = 0; i < n_images; i++)
    {
        free(tmp_layer[i].layer);
    }
    free(tmp_layer);
}

thread_workspace_t* alloc_workspace (int n_threads) {
    thread_workspace_t *p = calloc(n_threads, sizeof(thread_workspace_t));
    for (size_t i = 0; i < n_threads; i++)
    {
        p[i].training_image_buffer = malloc(784 * batch_size / 4 * sizeof(uint8_t));
        p[i].training_label_buffer = malloc(batch_size / 4 * sizeof(uint8_t));
        p[i].z_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].z_out = malloc(n_of_output_layer * sizeof(float));
        p[i].first_conv_layer_pre_activation = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1);
        p[i].second_conv_layer_pre_activation = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_width + 1);
        p[i].a_in = malloc(784 * sizeof(float));
        p[i].a_0 = calloc(n_of_input_layer, sizeof(float));
        p[i].a_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].a_out = malloc(n_of_output_layer * sizeof(float));
        p[i].first_conv_layer_activation = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1);
        p[i].second_conv_layer_activation = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_width + 1);
        p[i].first_maxpooling_layer = alloc_maxpool_layer(n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
        p[i].second_maxpooling_layer = alloc_maxpool_layer(n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_hight + 1)/2 - filter_width + 1)/2);
        p[i].delta_in = malloc(n_of_input_layer * sizeof(float));
        p[i].delta_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].delta_4 = malloc(n_of_output_layer * sizeof(float));
        p[i].backward_first_conv = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_hight + 1);
        p[i].backward_second_conv = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_hight + 1);
        p[i].backward_first_maxpool = alloc_maxpool_layer(n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_hight + 1)/2);
        p[i].backward_second_maxpool = alloc_maxpool_layer(n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2);
        p[i].grad_w1 = malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
        p[i].grad_w4 = malloc(n_of_first_hidden_layer * n_of_output_layer * sizeof(float));
        p[i].grad_b1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].grad_b4 = malloc(n_of_output_layer * sizeof(float));
        p[i].grad_to_b_conv1 = calloc(n_of_first_channel, sizeof(float));
        p[i].grad_to_b_conv2 = calloc(n_of_second_channel, sizeof(float));
        p[i].grad_to_first_conv_filter = alloc_filter(n_of_first_channel);
        p[i].grad_to_second_conv_filter = alloc_filter(n_of_first_channel * n_of_second_channel);
        p[i].grad_w1t = malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
        p[i].grad_w4t = malloc(n_of_first_hidden_layer * n_of_output_layer * sizeof(float));
        p[i].grad_b1t = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].grad_b4t = malloc(n_of_output_layer * sizeof(float));
        p[i].grad_to_b_conv1_t = calloc(n_of_first_channel, sizeof(float));
        p[i].grad_to_b_conv2_t = calloc(n_of_second_channel, sizeof(float));
        p[i].grad_to_first_conv_filter_t = alloc_filter(n_of_first_channel);
        p[i].grad_to_second_conv_filter_t = alloc_filter(n_of_first_channel * n_of_second_channel);
    }
    return p;
}

void free_workspace (thread_workspace_t *p, int n_threads) {
    for (size_t i = 0; i < n_threads; i++)
    {
        //buffer
        free(p[i].training_image_buffer); free(p[i].training_label_buffer);

        //pre-activations
        free(p[i].z_1); free(p[i].z_out); free_conv_layer(p[i].first_conv_layer_pre_activation, n_of_first_channel); free_conv_layer(p[i].second_conv_layer_pre_activation, n_of_second_channel);

        //activations
        free(p[i].a_in); free(p[i].a_0); free(p[i].a_1); free(p[i].a_out); free_conv_layer(p[i].first_conv_layer_activation, n_of_first_channel); free_conv_layer(p[i].second_conv_layer_activation, n_of_second_channel);
        
        //deltas
        free(p[i].delta_in); free(p[i].delta_1); free(p[i].delta_4); free_conv_layer(p[i].backward_first_conv, n_of_first_channel); free_conv_layer(p[i].backward_second_conv, n_of_second_channel); free_maxpool_layer(p[i].backward_first_maxpool, n_of_first_channel); free_maxpool_layer(p[i].backward_second_maxpool, n_of_second_channel);
        
        //gradients
        free(p[i].grad_w1); free(p[i].grad_w4); free(p[i].grad_b1); free(p[i].grad_b4); free(p[i].grad_to_b_conv1); free(p[i].grad_to_b_conv2); free_filter(p[i].grad_to_first_conv_filter, n_of_first_channel); free_filter(p[i].grad_to_second_conv_filter, n_of_first_channel * n_of_second_channel);
        
        //gradients total
        free(p[i].grad_w1t); free(p[i].grad_w4t); free(p[i].grad_b1t); free(p[i].grad_b4t); free(p[i].grad_to_b_conv1_t); free(p[i].grad_to_b_conv2_t); free_filter(p[i].grad_to_first_conv_filter_t, n_of_first_channel); free_filter(p[i].grad_to_second_conv_filter_t, n_of_first_channel * n_of_second_channel);
    }
    free(p);
}

void apply_dropout (float *operated_arr, bool *mask, int batch, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] = operated_arr[i] * mask[n_of_arr * batch + i] / (1 - dropout_rate);
    }
    
}

void generate_dropout_mask (bool *dropout_mask, int number_of_mask) {
    for (size_t i = 0; i < number_of_mask; i++)
    {
        if ((float)(rand())/RAND_MAX <= dropout_rate)
        {
            dropout_mask[i] = 0;
        }
        else
        {
            dropout_mask[i] = 1;
        }
    }
    
}

void add_weight (float *grad, float *weight, int n_of_grad) {
    for (size_t i = 0; i < n_of_grad; i++)
    {
        grad[i] += (regularization_rate * weight[i]);
    }
    
}

float f_arr_squared_sum (float *arr, int n_arr) {
    float sum = 0.0f;
    for (size_t i = 0; i < n_arr; i++)
    {
        sum += arr[i] * arr[i];
    }
    return sum;
}

void shuffle_indices(int* indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

float larger (float input_1, float input_2){
    if (input_1 > input_2) return input_1;
    else return input_2;
    
}

void add_bias (float *operated_arr, float *input_bias, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] = operated_arr[i] + input_bias[i];
    }
    
}

void vec_add_avx (const float *a, const float *b, float *c, int n) {
    __m256 va, vb, vc;

    for (size_t i = 0; i < n / 8; i++) {   
        va = _mm256_loadu_ps(&a[8*i]);
        vb = _mm256_loadu_ps(&b[8*i]);

        vc = _mm256_add_ps(va, vb);

        _mm256_storeu_ps(&c[8*i], vc);    
    }

    for (size_t i = 0; i < n % 8; i++) {
        c[(n / 8) * 8 + i] = a[(n / 8) * 8 + i] + b[(n / 8) * 8 + i];
    }        
}

int mmul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr){
    memset(output_arr, 0, n_of_output_arr * sizeof(float));
    
    for (int i = 0; i < n_of_output_arr; i++)
    {
        for (int j = 0; j < n_of_input_arr; j++)
        {
            output_arr[i] += matrix[n_of_input_arr * i + j] * input_arr[j];
        }
        
    }

    return 0;
}

float dot_product (const float *a, const float *b, int n) {
    __m256 va, vb, sum_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < n / 8; i++) {
        va = _mm256_loadu_ps(&a[8*i]);
        vb = _mm256_loadu_ps(&b[8*i]);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
    }

    // Horizontal reduction of sum_vec
    __m128 low = _mm256_castps256_ps128(sum_vec);
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // Handle remaining elements
    for (size_t i = (n / 8) * 8; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void mat_vec_mul(const float *A, const float *x, float *y, int M, int N) {
    for (size_t i = 0; i < M; i++)
    {
        y[i] = dot_product(&A[N*i], x, N);
    }
    
}

float extract_max (float *input_array, int n_of_input_arr){
    float max = input_array[0];
    for (int i = 0; i < n_of_input_arr; i++)
    {
        if (max < input_array[i])
        {
            max = input_array[i];
        }
        
    }
    
    return max;
}

void relu(float *input_arr, float *output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = larger(input_arr[i], 0.0f);
    }
    
}

void gelu (float *input_arr, float *output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = 0.5 * input_arr[i] * (1 + tanh(sqrt(2/PI) * (input_arr[i] + 0.044715 * pow(input_arr[i], 3))));
    }
}

void leaky_relu(float* input_arr, float* output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        if (input_arr[i] > 0.0f)
        {
            output_arr[i] = input_arr[i];
        }
        else
        {
            output_arr[i] = 0.01 * input_arr[i];
        }
    }
}

void softmax (float *input_arr, float *output_arr, int n_of_arr){
    float max = 0.0, sum = 0.0;
    float *tmp = malloc(n_of_arr * sizeof(float));
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        tmp[i] = exp(input_arr[i] - max);
        sum += tmp[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = tmp[j] / sum;
    }
    free(tmp);
}

void he_initialize_uniform(float *W, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        W[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
}

void compute_output_delta (float *output, float *output_layer, float *answer_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++) {
        output[i] = output_layer[i] - answer_arr[i];
    }
    
}

void weight_grad (float *z_delta, float *previous_activation_arr, float *output_arr, int n_of_output, int n_of_input){
    for (int i = 0; i < n_of_output; i++)
    {
        for (int j = 0; j < n_of_input; j++)
        {
            output_arr[i * n_of_input + j] = z_delta[i] * previous_activation_arr[j];
        }
        
    }
}

void grad_bias (float *input, float *output, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        output[i] = input[i];
    }
        
}

void compute_hidden_delta (float *z_delta, float *next_weight, float *z, float *output, int n_of_activation, int n_of_z_delta){
    memset(output, 0, n_of_activation * sizeof(float));
    for (int i = 0; i < n_of_activation; i++)
    {
        for (int j = 0; j < n_of_z_delta; j++)
        {
            output[i] += z_delta[j] * next_weight[j * n_of_activation+ i];
        }
        
    }

    for (int i = 0; i < n_of_activation; i++)
    {
        if (z[i] <= 0)
        {
            output[i] = 0.0f;
        }
        
    }
    
}

void compute_hidden_activation_delta (float *z_delta, float *next_weight, float *output, int n_of_activation, int n_of_z_delta){
    memset(output, 0, n_of_activation * sizeof(float));
    for (int i = 0; i < n_of_activation; i++)
    {
        for (int j = 0; j < n_of_z_delta; j++)
        {
            output[i] += z_delta[j] * next_weight[j * n_of_activation+ i];
        }
        
    }
    
}

void compute_hidden_delta_leaky (float *z_delta, float *next_weight, float *z, float *output, int n_of_activation, int n_of_z_delta){
    for (int i = 0; i < n_of_activation; i++)
    {
        output[i] = 0.0f;
    }
    for (int i = 0; i < n_of_activation; i++)
    {
        for (int j = 0; j < n_of_z_delta; j++)
        {
            output[i] += z_delta[j] * next_weight[j * n_of_activation+ i];
        }
        
    }

    for (int i = 0; i < n_of_activation; i++)
    {
        if (z[i] <= 0)
        {
            output[i] = 0.01 * output[i];
        }
        
    }
    
}

void momentum_update(float *weight, float *weight_grad, float *velocity_weight_grad, float *bias, float *bias_grad, float *velocity_bias_grad, int n_of_weight, int n_of_bias){
    for (int i = 0; i < n_of_weight; i++)
    {
        velocity_weight_grad[i] = momentum_beta * velocity_weight_grad[i] + weight_grad[i];
        weight[i] -= learning_rate * velocity_weight_grad[i];
        weight_grad[i] = 0.0f;
    }
    for (int i = 0; i < n_of_bias; i++)
    {
        velocity_bias_grad[i] = momentum_beta * velocity_bias_grad[i] + bias_grad[i];
        bias[i] -= learning_rate * velocity_bias_grad[i];
        bias_grad[i] = 0.0f;
    }
}

void momentum_update_conv (conv_filter_t *filter, conv_filter_t *filter_grad, conv_filter_t *velocity_filter_grad, float *bias, float *bias_grad, float *velocity_bias_grad, int n_of_filter, int n_of_bias){
    for (size_t c = 0; c < n_of_filter; c++)
    {
        for (size_t h = 0; h < filter_hight; h++)
        {
            for (size_t w = 0; w < filter_width; w++)
            {
                velocity_filter_grad[c].filter[h * filter_width + w] = momentum_beta * velocity_filter_grad[c].filter[h * filter_width + w] + filter_grad[c].filter[h * filter_width + w];
                filter[c].filter[h * filter_width + w] -= learning_rate * velocity_filter_grad[c].filter[h * filter_width + w];
                filter_grad[c].filter[h * filter_width + w] = 0.0f;
            }
            
        }
        
    }
    for (int i = 0; i < n_of_bias; i++)
    {
        velocity_bias_grad[i] = momentum_beta * velocity_bias_grad[i] + bias_grad[i];
        bias[i] -= learning_rate * velocity_bias_grad[i];
        bias_grad[i] = 0.0f;
    }
}

int find_max_index (float *arr, int size){
    if(size <= 0){
        return -1;
    }
    int max_index = 0;
    float max_value = arr[0];

    for (int i = 0; i < size; i++)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
            max_index = i;
        }
        
    }
    
    return max_index;
}

void convolution_single_to_multi(float *input_layer, conv_filter_t *f, conv_layer_t *l, int n_input_hight, int n_input_width, int n_output_channel) {
    // Calculations for dimensions
    // 変数名は元のコードのスペルミス(hight)を踏襲しています
    int output_hight = n_input_hight - filter_hight + 1;
    int output_width = n_input_width - filter_width + 1;
    int output_size = output_hight * output_width;

    // 各出力チャンネルごとに処理
    for (int oc = 0; oc < n_output_channel; oc++) {
        // ポインタを一度だけ取得（ループ内での構造体アクセス回避）
        float *restrict out_ptr = l[oc].layer;
        const float *restrict filter_ptr = f[oc].filter;

        // ゼロ初期化
        // memsetや単純なループで展開可能。コンパイラが最適化しやすい形にします。
        for (int i = 0; i < output_size; i++) {
            out_ptr[i] = 0.0f;
        }

        // 畳み込み処理
        // 元のコード: 出力ピクセル(oh,ow) -> フィルタ
        // 最適化後: フィルタ -> 出力ピクセル(oh,ow)
        // これにより、out_ptrとinput_layerへのアクセスが連続メモリアクセスになり、
        // SIMD命令による並列化が可能になります。
        
        for (int kh = 0; kh < filter_hight; kh++) {
            int input_row_offset = kh * n_input_width;
            int filter_row_offset = kh * filter_width;

            for (int kw = 0; kw < filter_width; kw++) {
                float weight = filter_ptr[filter_row_offset + kw];
                
                // 入力画像の開始位置（kwとkhのオフセットを適用）
                const float *restrict in_ptr = input_layer + input_row_offset + kw;

                // 行ごとのスキャン
                for (int oh = 0; oh < output_hight; oh++) {
                    int out_row_start = oh * output_width;
                    int in_row_start = oh * n_input_width;
                    
                    // この内側のループがベクトル化（SIMD化）の対象になります
                    // out_ptr[row_start + ow] += weight * in_ptr[in_row_start + ow];
                    // という処理をポインタ演算で記述します。
                    
                    float *restrict out_row = out_ptr + out_row_start;
                    const float *restrict in_row = in_ptr + in_row_start;

                    for (int ow = 0; ow < output_width; ow++) {
                        out_row[ow] += weight * in_row[ow];
                    }
                }
            }
        }
    }
}

/**
 * @param input conv_layer_tではなくmaxpool_layer_tであることに注意
 */
/**
 * @param input conv_layer_tではなくmaxpool_layer_tであることに注意
 */
void convolution_multi_to_multi(maxpool_layer_t *input, conv_filter_t *filter, conv_layer_t *output, int n_input_hight, int n_input_width, int n_input_channel, int n_output_channel) {
    // Dimensions
    int n_output_hight = n_input_hight - filter_hight + 1;
    int n_output_width = n_input_width - filter_width + 1;
    int n_output_size = n_output_hight * n_output_width;

    // 出力チャンネルごとのループ
    for (int oc = 0; oc < n_output_channel; oc++) {
        // ポインタのキャッシュ（構造体アクセスの削減）
        float *restrict out_ptr = output[oc].layer;

        // ゼロ初期化
        for (int i = 0; i < n_output_size; i++) {
            out_ptr[i] = 0.0f;
        }

        // 入力チャンネルごとの累積ループ
        for (int ic = 0; ic < n_input_channel; ic++) {
            // フィルタと入力のポインタを取得
            // 元のコード: filter[n_input_channel * oc + ic] のアクセス順序を維持
            const float *restrict filter_ptr = filter[n_input_channel * oc + ic].filter;
            const float *restrict in_ptr = input[ic].layer;

            // フィルタの高さ・幅のループ（外側へ移動）
            for (int kh = 0; kh < filter_hight; kh++) {
                int filter_row_offset = kh * filter_width;
                
                for (int kw = 0; kw < filter_width; kw++) {
                    float weight = filter_ptr[filter_row_offset + kw];

                    // 入力画像における開始位置のオフセット
                    // input[ (oh + kh) * width + (ow + kw) ] となるように調整
                    const float *restrict in_start = in_ptr + kh * n_input_width + kw;

                    // 画像の高さ・幅のループ（内側へ移動）
                    // ここでSIMD並列化が効きます
                    for (int oh = 0; oh < n_output_hight; oh++) {
                        float *restrict out_row = out_ptr + oh * n_output_width;
                        const float *restrict in_row = in_start + oh * n_input_width;

                        for (int ow = 0; ow < n_output_width; ow++) {
                            // 連続したメモリへのアクセス
                            out_row[ow] += weight * in_row[ow];
                        }
                    }
                }
            }
        }
    }
}

void maxpool(conv_layer_t *input, maxpool_layer_t *output, int channels, int in_h, int in_w, int pool_size) {
    //standby
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t oh = 0; oh < out_h; oh++)
        {
            for (size_t ow = 0; ow < out_w; ow++)
            {
                float max = -__FLT_MAX__;
                int max_indics;
                for (size_t kh = 0; kh < pool_size; kh++)
                {
                    for (size_t kw = 0; kw < pool_size; kw++)
                    {
                        float value = input[c].layer[(oh*pool_size+kh)*in_w + (ow*pool_size+kw)];
                        if (value > max)
                        {
                            max = value;
                            max_indics = pool_size * kh + kw;
                        }
                        
                    }
                    
                }
                output[c].layer[out_w * oh + ow] = max;
                output[c].mask[out_w * oh + ow] = max_indics;
            }
            
        }
        
    }
    
}

void maxpool_backward(maxpool_layer_t *d_output, conv_layer_t *d_input, maxpool_layer_t *mask_store, int n_channels, int in_h, int in_w, int pool_size) {
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;

    // d_inputをゼロ初期化
    for (int c = 0; c < n_channels; c++)
        memset(d_input[c].layer, 0, in_h * in_w * sizeof(float));

    for (int c = 0; c < n_channels; c++)
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++) {
                int idx = mask_store[c].mask[oh * out_w + ow];
                int kh  = idx / pool_size;
                int kw  = idx % pool_size;
                // max位置だけに勾配を流す
                d_input[c].layer[(oh*pool_size+kh)*in_w + (ow*pool_size+kw)]
                    += d_output[c].layer[oh * out_w + ow];
            }
}

void flatten(float *output_arr, maxpool_layer_t *input_conv_layer, int in_channel, int in_hight, int in_width) {
    for (size_t c = 0; c < in_channel; c++)
    {
        for (size_t h = 0; h < in_hight; h++)
        {
            for (size_t w = 0; w < in_width; w++)
            {
                output_arr[in_hight * in_width * c +  in_width * h + w] = input_conv_layer[c].layer[in_width * h + w];
            }
            
        }
        
    }
    

}

void unflatten (maxpool_layer_t *output_layer, float *input_arr, int c, int in_h, int in_w) {
    for (size_t channel = 0; channel < c;channel++)
    {
        for (size_t h = 0; h < in_h; h++)
        {
            for (size_t w = 0; w < in_w; w++)
            {
                output_layer[channel].layer[in_w * h + w] = input_arr[in_h * in_w *  channel +  in_w * h + w];
            }
            
        }
        
    }
    
}

void add_bias_conv (conv_layer_t *layer, float *bias, int in_channel, int in_hight, int in_width) {
    for (size_t c = 0; c < in_channel; c++)
    {
        for (size_t h = 0; h < in_hight; h++)
        {
            for (size_t w = 0; w < in_width; w++)
            {
                layer[c].layer[in_width * h + w] += bias[c];
            }
            
        }
        
    }
    
}

void backward_relu (float *output_arr, float *input_arr, float *pre_activation, int n_arr) {
    for (size_t i = 0; i < n_arr; i++)
    {
        if (pre_activation[i] > 0)
        {
            output_arr[i] = input_arr[i];
        }
        else
        {
            output_arr[i] = 0;
        }
        
        
    }
    
}

float float_array_sum  (float *arr, int n_arr) {
    float sum = 0.0f;
    for (size_t i = 0; i < n_arr; i++)
    {
        sum += arr[i];
    }
    return sum;
}

/**
 * @param in_h 画像高さは畳み込み処理前のものを入力
 * @param in_w 画像幅は畳み込み処理前のものを入力
 */
void backward_conv_filter_single_to_multi (conv_filter_t *output_grad, float *activation, conv_layer_t *z_delta, int in_channel, int in_h, int in_w) {
// 出力マップのサイズ（フィルタサイズはグローバル変数と仮定）
    int out_h = in_h - filter_hight + 1;
    int out_w = in_w - filter_width + 1;

    // 1. フィルタ勾配のゼロ初期化（memset で高速化）
    for (int c = 0; c < in_channel; ++c) {
        memset(output_grad[c].filter, 0,
               filter_hight * filter_width * sizeof(float));
    }

    // 2. メイン計算：ループ順序の入れ替え + ブロッキング
    for (int c = 0; c < in_channel; ++c) {
        float *filter = output_grad[c].filter;
        float *z = z_delta[c].layer;          // 出力誤差 (out_h x out_w)

        // 出力マップをブロック単位で処理
        for (int hb = 0; hb < out_h; hb += BLOCK_H) {
            int h_end = (hb + BLOCK_H < out_h) ? hb + BLOCK_H : out_h;
            for (int wb = 0; wb < out_w; wb += BLOCK_W) {
                int w_end = (wb + BLOCK_W < out_w) ? wb + BLOCK_W : out_w;

                // ブロック内の全出力位置を処理
                for (int h = hb; h < h_end; ++h) {
                    for (int w = wb; w < w_end; ++w) {
                        float delta = z[out_w * h + w];  // 出力誤差値

                        // この出力位置に対応するフィルタ領域を更新
                        for (int kh = 0; kh < filter_hight; ++kh) {
                            for (int kw = 0; kw < filter_width; ++kw) {
                                // 入力活性化の対応位置
                                float act = activation[in_w * (kh + h) + (kw + w)];
                                filter[filter_width * kh + kw] += delta * act;
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @param in_h 画像高さは畳み込み処理前のものを入力
 * @param in_w 画像幅は畳み込み処理前のものを入力
 * @param out_channel 順伝播での出力チャネル数
 * @param in_channel 順伝播での入力チャネル数
 */
void backward_conv_filter_multi_to_multi (conv_filter_t *output_grad, maxpool_layer_t *activation, conv_layer_t *z_delta, int out_channel, int in_channel, int in_h, int in_w) {
    int out_h = in_h - filter_hight + 1;
    int out_w = in_w - filter_width + 1;

    // 1. フィルタ勾配のゼロ初期化（memset で高速化）
    for (int oc = 0; oc < out_channel; ++oc) {
        for (int ic = 0; ic < in_channel; ++ic) {
            memset(output_grad[in_channel * oc + ic].filter, 0,
                   filter_hight * filter_width * sizeof(float));
        }
    }

    // 2. メイン計算：ループ順序の入れ替え + ブロッキング
    for (int oc = 0; oc < out_channel; ++oc) {
        float *z = z_delta[oc].layer;          // 出力誤差 (out_h x out_w)

        // 出力マップをブロック単位で処理
        for (int hb = 0; hb < out_h; hb += BLOCK_H) {
            int h_end = (hb + BLOCK_H < out_h) ? hb + BLOCK_H : out_h;
            for (int wb = 0; wb < out_w; wb += BLOCK_W) {
                int w_end = (wb + BLOCK_W < out_w) ? wb + BLOCK_W : out_w;

                // ブロック内の全出力位置を処理
                for (int h = hb; h < h_end; ++h) {
                    for (int w = wb; w < w_end; ++w) {
                        float delta = z[out_w * h + w];  // 出力誤差値

                        // この出力位置に対応する全入力チャネル・フィルタ位置を更新
                        for (int ic = 0; ic < in_channel; ++ic) {
                            float *filter = output_grad[in_channel * oc + ic].filter;
                            float *act = activation[ic].layer;   // 入力活性化

                            for (int kh = 0; kh < filter_hight; ++kh) {
                                for (int kw = 0; kw < filter_width; ++kw) {
                                    // 入力活性化の対応位置
                                    float act_val = act[in_w * (kh + h) + (kw + w)];
                                    filter[filter_width * kh + kw] += delta * act_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void backward_conv_layer (maxpool_layer_t *d_input, conv_layer_t *d_z, conv_filter_t *filter, int n_input_channel, int n_output_channel, int in_h, int in_w) {
    int out_h = in_h - filter_hight + 1;
    int out_w = in_w - filter_width + 1;

    // dx: 入力と同じ形状 [ic].layer[ih * in_w + iw]

    // ゼロ初期化
    for (int ic = 0; ic < n_input_channel; ic++)
    memset(d_input[ic].layer, 0, in_h * in_w * sizeof(float));

    for (int oc = 0; oc < n_output_channel; oc++)
        for (int ic = 0; ic < n_input_channel; ic++)
            for (int kh = 0; kh < filter_hight; kh++)
                for (int kw = 0; kw < filter_width; kw++)
                    for (int oh = 0; oh < out_h; oh++)
                        for (int ow = 0; ow < out_w; ow++)
                            d_input[ic].layer[(oh + kh) * in_w + (ow + kw)] += d_z[oc].layer[oh * out_w + ow]* filter[n_input_channel * oc + ic].filter[filter_width * kh + kw];
}

void* training_threaded (void* arg){
    //standby section
    thread_workspace_t* datas = (thread_workspace_t*)arg;
    int answer;
    float answer_arr[10];
    float loss;
    
    //compute section
        for (int loop = 0; loop < batch_size / num_threads; loop++){

            //inputting data
            for (int i = 0; i < n_of_input_layer; i++){
                datas->a_in[i] = (float)(datas->training_image_buffer[784 * loop + i])/255;
            }

            //inputting label
            answer = datas->training_label_buffer[loop];

            //forward pass
            convolution_single_to_multi(datas->a_in, datas->first_conv_filter, datas->first_conv_layer_pre_activation, 28, 28, n_of_first_channel);
            add_bias_conv(datas->first_conv_layer_pre_activation, datas->first_conv_bias, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1));
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                relu(datas->first_conv_layer_pre_activation[i].layer, datas->first_conv_layer_activation[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            maxpool(datas->first_conv_layer_activation, datas->first_maxpooling_layer, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1), 2);
            
            convolution_multi_to_multi(datas->first_maxpooling_layer, datas->second_conv_filter, datas->second_conv_layer_pre_activation, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2, n_of_first_channel, n_of_second_channel);
            add_bias_conv(datas->second_conv_layer_pre_activation, datas->second_conv_bias, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1));
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                relu(datas->second_conv_layer_pre_activation[i].layer, datas->second_conv_layer_activation[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            maxpool(datas->second_conv_layer_activation, datas->second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1), 2);

            flatten(datas->a_0, datas->second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_width + 1)/2 - filter_width + 1)/2);
            
            mmul(datas->z_1, datas->a_0, datas->w1, n_of_first_hidden_layer, n_of_input_layer);
            add_bias(datas->z_1, datas->b1, n_of_first_hidden_layer);

            relu(datas->z_1, datas->a_1, n_of_first_hidden_layer);

            mmul(datas->z_out, datas->a_1, datas->wout, n_of_output_layer, n_of_first_hidden_layer);
            add_bias(datas->z_out, datas->bout, n_of_output_layer);

            softmax(datas->z_out, datas->a_out, n_of_output_layer);


            //loss function (cross entropy)
            for (int i = 0; i < n_of_output_layer; i++){
                answer_arr[i] = 0.0;
            }
            answer_arr[answer] = 1.0;
            loss = 0.0f;
            for (int i = 0; i < n_of_output_layer; i++){
                loss += answer_arr[i] * logf(datas->a_out[i] + 1e-8f);
            }
            loss = -loss;

            //backward pass
            compute_output_delta(datas->delta_4, datas->a_out, answer_arr, n_of_output_layer);

            weight_grad(datas->delta_4, datas->a_1, datas->grad_w4, n_of_output_layer, n_of_first_hidden_layer);
            grad_bias(datas->delta_4, datas->grad_b4, n_of_output_layer);

            compute_hidden_delta(datas->delta_4, datas->wout, datas->z_1, datas->delta_1, n_of_first_hidden_layer, n_of_output_layer);

            weight_grad(datas->delta_1, datas->a_0, datas->grad_w1, n_of_first_hidden_layer, n_of_input_layer);
            grad_bias(datas->delta_1, datas->b1, n_of_first_hidden_layer);

            compute_hidden_activation_delta(datas->delta_1, datas->w1, datas->delta_in, n_of_input_layer, n_of_first_hidden_layer);

            unflatten(datas->backward_second_maxpool, datas->delta_in, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_width + 1)/2 - filter_width + 1)/2);
            
            maxpool_backward(datas->backward_second_maxpool, datas->backward_second_conv, datas->second_maxpooling_layer, n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_width + 1)/2 - filter_width + 1, 2);
            
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                backward_relu(datas->backward_second_conv[i].layer, datas->backward_second_conv[i].layer, datas->second_conv_layer_pre_activation[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            //バイアス勾配
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                datas->grad_to_b_conv2[i] = float_array_sum(datas->backward_second_conv[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            //フィルター勾配
            backward_conv_filter_multi_to_multi(datas->grad_to_second_conv_filter, datas->first_maxpooling_layer, datas->backward_second_conv, n_of_second_channel, n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
            

            //前層アクティベーション勾配
            backward_conv_layer(datas->backward_first_maxpool, datas->backward_second_conv, datas->second_conv_filter, n_of_first_channel, n_of_second_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
            
            maxpool_backward(datas->backward_first_maxpool, datas->backward_first_conv, datas->first_maxpooling_layer, n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1, 2);
            
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                backward_relu(datas->backward_first_conv[i].layer, datas->backward_first_conv[i].layer, datas->first_conv_layer_pre_activation[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            //バイアス勾配
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                datas->grad_to_b_conv1[i] = float_array_sum(datas->backward_first_conv[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            //フィルター勾配
            backward_conv_filter_single_to_multi(datas->grad_to_first_conv_filter, datas->a_in, datas->backward_first_conv, n_of_first_channel, 28, 28);

            for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
            {
                datas->grad_w1t[i] += datas->grad_w1[i];
            }
            for (size_t i = 0; i < n_of_first_hidden_layer; i++)
            {
                datas->grad_b1t[i] += datas->grad_b1[i];
            }
            for (size_t i = 0; i < n_of_first_hidden_layer * n_of_output_layer; i++)
            {
                datas->grad_w4t[i] += datas->grad_w4[i];
            }
            for (size_t i = 0; i < n_of_output_layer; i++)
            {
                datas->grad_b4t[i] += datas->grad_b4[i];
            }
            for (size_t c = 0; c < n_of_first_channel; c++)
            {
                for (size_t h = 0; h < filter_hight; h++)
                {
                    for (size_t w = 0; w < filter_width; w++)
                    {
                        datas->grad_to_first_conv_filter_t[c].filter[h * filter_width + w] += datas->grad_to_first_conv_filter[c].filter[h * filter_width + w];
                    }
                    
                }
                
            }
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                datas->grad_to_b_conv1_t[i] += datas->grad_to_b_conv1[i];
            }
            for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
            {
                for (size_t h = 0; h < filter_hight; h++)
                {
                    for (size_t w = 0; w < filter_width; w++)
                    {
                        datas->grad_to_second_conv_filter_t[c].filter[h * filter_width + w] += datas->grad_to_second_conv_filter[c].filter[h * filter_width + w];
                    }
                    
                }
                
            }
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                datas->grad_to_b_conv2_t[i] += datas->grad_to_b_conv2[i];
            }

        }

        pthread_mutex_lock(&mutex);
        for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
        {
            datas->return_grad_w1t[i] += datas->grad_w1t[i];
        }
        for (size_t i = 0; i < n_of_first_hidden_layer; i++)
        {
            datas->return_grad_b1t[i] += datas->grad_b1t[i];
        }
        for (size_t i = 0; i < n_of_first_hidden_layer * n_of_output_layer; i++)
        {
            datas->return_grad_w4t[i] += datas->grad_w4t[i];
        }
        for (size_t i = 0; i < n_of_output_layer; i++)
        {
            datas->return_grad_b4t[i] += datas->grad_b4t[i];
        }
        for (size_t c = 0; c < n_of_first_channel; c++)
        {
            for (size_t h = 0; h < filter_hight; h++)
            {
                for (size_t w = 0; w < filter_width; w++)
                {
                    datas->return_grad_to_first_conv_filter_t[c].filter[h * filter_width + w] += datas->grad_to_first_conv_filter_t[c].filter[h * filter_width + w];
                }
                    
            }
                
        }
        for (size_t i = 0; i < n_of_first_channel; i++)
        {
            datas->return_grad_to_b_conv1_t[i] += datas->grad_to_b_conv1_t[i];
        }
        for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
        {
            for (size_t h = 0; h < filter_hight; h++)
            {
                for (size_t w = 0; w < filter_width; w++)
                {
                    datas->return_grad_to_second_conv_filter_t[c].filter[h * filter_width + w] += datas->grad_to_second_conv_filter_t[c].filter[h * filter_width + w];
                }
                    
            }
                
        }
        for (size_t i = 0; i < n_of_second_channel; i++)
        {
            datas->return_grad_to_b_conv2_t[i] += datas->grad_to_b_conv2_t[i];
        }
        pthread_mutex_unlock(&mutex);
        return NULL;
}

int main (void){
    srand(time(NULL));
    //define variables
    int answer = 0, hit = 0;
    float answer_arr[n_of_output_layer] = {0.0f};
    float loss = 0.0f, avg_loss = 0.0f;
    int *order_indices = calloc(60000, sizeof(int));
    for (int i = 0; i < 60000; i++)
    {
        order_indices[i] = i;
    }
    int adam_t = 0;
    clock_t start, end;
    pthread_t th[num_threads];

    
    //define pointer
    float *input_layer;
    float *first_hidden_layer;
    float *output_layer;
    float *weight_to_first_hidden_layer;
    float *bias_of_first_hidden_layer;
    float *weight_to_output_layer;
    float *bias_of_output_layer;
    float *z1;
    float *zout;
    float *delta_4, *delta_1;
    float *grad_to_w4, *grad_to_b4, *grad_to_w1, *grad_to_b1;
    float *m_w1, *m_w4;
    float *m_b1, *m_b4;
    float *v_w1, *v_w4;
    float *v_b1, *v_b4;
    float grad_to_b_conv1[n_of_first_channel];
    float grad_to_b_conv2[n_of_second_channel];

    //allocetion params
    float *input_image = calloc(784, sizeof(float));
    input_layer = (float*)malloc(n_of_input_layer * sizeof(float));
    first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    weight_to_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * n_of_input_layer * sizeof(float));
    bias_of_first_hidden_layer = (float*)calloc(n_of_first_hidden_layer, sizeof(float));
    weight_to_output_layer = (float*)malloc(n_of_output_layer * n_of_first_hidden_layer * sizeof(float));
    bias_of_output_layer = (float*)calloc(n_of_output_layer, sizeof(float));
    z1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    zout = (float*)malloc(n_of_output_layer * sizeof(float));
    delta_4 = (float*)malloc(n_of_output_layer * sizeof(float));
    delta_1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    float *delta_in = calloc(n_of_input_layer, sizeof(float));
    grad_to_w4 = (float*)malloc(n_of_first_hidden_layer * n_of_output_layer * sizeof(float));
    grad_to_w1 = (float*)malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
    grad_to_b4 = (float*)malloc(n_of_output_layer * sizeof(float));
    grad_to_b1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    conv_layer_t *first_conv_layer_pre_activation = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1);
    conv_layer_t *first_conv_layer_activation = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1);
    conv_filter_t *first_conv_filter = alloc_filter(n_of_first_channel);
    maxpool_layer_t *first_maxpooling_layer = alloc_maxpool_layer(n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
    conv_layer_t *second_conv_layer_pre_activation = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_width + 1);
    conv_layer_t *second_conv_layer_activation = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_width + 1);
    conv_filter_t *second_conv_filter = alloc_filter(n_of_first_channel * n_of_second_channel);
    maxpool_layer_t *second_maxpooling_layer = alloc_maxpool_layer(n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_hight + 1)/2 - filter_width + 1)/2);
    float *first_conv_bias = calloc(n_of_first_channel, sizeof(float));
    float *second_conv_bias = calloc(n_of_second_channel, sizeof(float));
    maxpool_layer_t *backward_second_maxpool = alloc_maxpool_layer(n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2);
    conv_layer_t *backward_second_conv = alloc_conv_layer(n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_hight + 1)/2 - filter_hight + 1);
    conv_filter_t *grad_to_second_conv_filter = alloc_filter(n_of_first_channel * n_of_second_channel);
    maxpool_layer_t *backward_first_maxpool = alloc_maxpool_layer(n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_hight + 1)/2);
    conv_layer_t *backward_first_conv = alloc_conv_layer(n_of_first_channel, 28 - filter_hight + 1, 28 - filter_hight + 1);
    conv_filter_t *grad_to_first_conv_filter = alloc_filter(n_of_first_channel);

    //total buffer
    float *grad_to_w4t = calloc(n_of_first_hidden_layer * n_of_output_layer, sizeof(float));
    float *grad_to_w1t = calloc(n_of_input_layer * n_of_first_hidden_layer, sizeof(float));
    float *grad_to_b4t = calloc(n_of_output_layer, sizeof(float));
    float *grad_to_b1t = calloc(n_of_first_hidden_layer, sizeof(float));
    conv_filter_t *grad_to_first_conv_filter_t = alloc_filter(n_of_first_channel);
    conv_filter_t *grad_to_second_conv_filter_t = alloc_filter(n_of_first_channel * n_of_second_channel);
    float *grad_to_b_conv1_t = calloc(n_of_first_channel, sizeof(float));
    float *grad_to_b_conv2_t = calloc(n_of_second_channel, sizeof(float));

    //momentum buffer
    float *velocity_grad_buffer_w1t = calloc(n_of_input_layer * n_of_first_hidden_layer, sizeof(float));
    float *velocity_grad_buffer_w4t = calloc(n_of_first_hidden_layer * n_of_output_layer, sizeof(float));
    float *velocity_grad_buffer_b1t = calloc(n_of_first_hidden_layer, sizeof(float));
    float *velocity_grad_buffer_b4t = calloc(n_of_output_layer, sizeof(float));
    conv_filter_t *velocity_grad_buffer_f1t = alloc_filter(n_of_first_channel);
    conv_filter_t *velocity_grad_buffer_f2t = alloc_filter(n_of_first_channel * n_of_second_channel);
    float *velocity_grad_buffer_conv_b1t = calloc(n_of_first_channel, sizeof(float));
    float *velocity_grad_buffer_conv_b2t = calloc(n_of_second_channel, sizeof(float));

    //thread_workspace_t *ws = alloc_workspace(4);
    bool *dropout_mask_for_first_hidden_layer = malloc(n_of_first_hidden_layer * (60000/batch_size));

    //file
    FILE *learning_data_images, *learning_data_labels, *test_data_images, *test_data_labels, *fp;

    //thread workspace alloc
    thread_workspace_t *ws = alloc_workspace(num_threads);

    //weight initialize
    he_initialize_uniform(weight_to_first_hidden_layer, n_of_input_layer, n_of_first_hidden_layer);
    he_initialize_uniform(weight_to_output_layer, n_of_first_hidden_layer, n_of_output_layer);
    // 第1畳み込み層の初期化 (fan_in = 1 * filter_hight * filter_width)
    float limit1 = sqrtf(6.0f / (1.0f * filter_hight * filter_width));
    for (int i = 0; i < n_of_first_channel; i++) {
        for (int j = 0; j < filter_hight * filter_width; j++) {
            first_conv_filter[i].filter[j] = ((float)rand() / RAND_MAX) * 2.0f * limit1 - limit1;
        }
    }

    // 第2畳み込み層の初期化 (fan_in = n_of_first_channel * filter_hight * filter_width)
    float limit2 = sqrtf(6.0f / (n_of_first_channel * filter_hight * filter_width));
    for (int i = 0; i < n_of_first_channel * n_of_second_channel; i++) {
        for (int j = 0; j < filter_hight * filter_width; j++) {
            second_conv_filter[i].filter[j] = ((float)rand() / RAND_MAX) * 2.0f * limit2 - limit2;
        }
    }

    //loading datas
    learning_data_images = fopen(train_images, "rb");
    if (learning_data_images == NULL)
    {
        printf("train images err\n");
        return 1;
    }
    printf("train images have loaded successfully.\n");
    learning_data_labels = fopen(train_labels, "rb");
    if (learning_data_labels == NULL)
    {
        printf("train labels err\n");
        return 2;
    }
    printf("train labels have loaded successfully\n");
    test_data_images = fopen(test_images, "rb");
    if (test_data_images == NULL)
    {
        printf("test images err\n");
        return 3;
    }
    printf("test images have loaded successfully.\n");
    test_data_labels = fopen(test_labels, "rb");
    if (test_data_labels == NULL)
    {
        printf("test labels err\n");
        return 4;
    }
    printf("test labels have loaded successfully\n");
    fp = fopen("test.csv", "w");
    fprintf(fp, ",train loss,test loss, hit rate\n");

    //offset data
    fseek(test_data_images, 16, 0);
    fseek(test_data_labels, 8, 0);

    memset(grad_to_w1, 0, n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
    memset(grad_to_w4, 0, n_of_first_hidden_layer * n_of_output_layer * sizeof(float));
    memset(grad_to_b1, 0, n_of_first_hidden_layer * sizeof(float));
    memset(grad_to_b4, 0, n_of_output_layer * sizeof(float));
    for (int i = 0; i < n_of_first_channel; i++)
        memset(grad_to_first_conv_filter[i].filter, 0, filter_hight * filter_width * sizeof(float));
    for (int i = 0; i < n_of_first_channel * n_of_second_channel; i++)
        memset(grad_to_second_conv_filter[i].filter, 0, filter_hight * filter_width * sizeof(float));

    printf("Training start\n");
    for (int epoch_loop = 0; epoch_loop < epoch; epoch_loop++){
        avg_loss = 0.0f;
        hit = 0;
        shuffle_indices(order_indices, 60000);

        //learning section
        if (threaded == 1)
        {
            //standby section
            for (size_t i = 0; i < num_threads; i++)
            {
                ws[i].w1 = weight_to_first_hidden_layer;
                ws[i].wout = weight_to_output_layer;
                ws[i].b1 = bias_of_first_hidden_layer;
                ws[i].bout = bias_of_output_layer;
                ws[i].first_conv_filter = first_conv_filter;
                ws[i].first_conv_bias = first_conv_bias;
                ws[i].second_conv_filter = second_conv_filter;
                ws[i].second_conv_bias = second_conv_bias;
                ws[i].return_grad_b1t = grad_to_b1t;
                ws[i].return_grad_b4t = grad_to_b4t;
                ws[i].return_grad_to_b_conv1_t = grad_to_b_conv1_t;
                ws[i].return_grad_to_b_conv2_t = grad_to_b_conv2_t;
                ws[i].return_grad_to_first_conv_filter_t = grad_to_first_conv_filter_t;
                ws[i].return_grad_to_second_conv_filter_t = grad_to_second_conv_filter_t;
                ws[i].return_grad_w1t = grad_to_w1t;
                ws[i].return_grad_w4t = grad_to_w4t;
            }

            //file buffering
            uint8_t *training_image_buffer = malloc(60000 * 784 * sizeof(uint8_t));
            uint8_t *training_label_buffer = malloc(60000 * sizeof(uint8_t));
            fseek(learning_data_images, 16, SEEK_SET);
            fread(training_image_buffer, sizeof(uint8_t), 60000 * 784, learning_data_images);
            fseek(learning_data_labels, 8, SEEK_SET);
            fread(training_label_buffer, sizeof(uint8_t), 60000, learning_data_labels);

            //training section
            for (int loop = 0; loop < 60000/batch_size; loop++)
            {
                
                if (!(loop%100) && debug) {printf("%d datas have been processed.\n", loop);}

                //reset total grad
                for (size_t i = 0; i < num_threads; i++)
                {
                    for (size_t j = 0; j < n_of_input_layer * n_of_first_hidden_layer; j++)
                    {
                        ws[i].grad_w1t[j] = 0.0f;
                    }

                } 
                for (size_t i = 0; i < num_threads; i++)
                {
                    for (size_t j = 0; j < n_of_first_hidden_layer * n_of_output_layer; j++)
                    {
                        ws[i].grad_w4t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < num_threads; i++)
                {
                    for (size_t j = 0; j < n_of_first_hidden_layer; j++)
                    {
                        ws[i].grad_b1t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < num_threads; i++)
                {
                    for (size_t j = 0; j < n_of_output_layer; j++)
                    {
                        ws[i].grad_b4t[j] = 0.0f;
                    }
                
                }

                //datas inport
                for (size_t i = 0; i < num_threads; i++)
                {
                    for (size_t j = 0; j < batch_size / num_threads; j++)
                    {
                        for (size_t copy_image = 0; copy_image < 784; copy_image++)
                        {
                            ws[i].training_image_buffer[784 * j + copy_image] = training_image_buffer[784 * order_indices[(60000/num_threads) * i + j + batch_size / num_threads * loop] + copy_image];
                        }
                        
                        ws[i].training_label_buffer[j] = training_label_buffer[order_indices[(60000/num_threads) * i + j + batch_size / num_threads * loop]];
                    }
                    
                }
                
                //compute grad
                for (size_t i = 0; i < num_threads; i++)
                {
                    pthread_create(&th[i], NULL, training_threaded, &ws[i]);
                }
                
                //wait
                for (size_t i = 0; i < num_threads; i++)
                {
                    pthread_join(th[i], NULL);
                }
                

                //update params
                //average
                for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
                {
                    grad_to_w1t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_first_hidden_layer; i++)
                {
                    grad_to_b1t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_first_hidden_layer * n_of_output_layer; i++)
                {
                    grad_to_w4t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_output_layer; i++)
                {
                    grad_to_b4t[i] /= batch_size;
                }
                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_first_conv_filter_t[c].filter[h * filter_width + w] /= batch_size;
                        }
                        
                    }
                    
                }
                for (size_t i = 0; i < n_of_first_channel; i++)
                {
                    grad_to_b_conv1_t[i] /= batch_size;
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_second_conv_filter_t[c].filter[h * filter_width + w] /= batch_size;
                        }
                    
                    }
                
                }
                for (size_t i = 0; i < n_of_second_channel; i++)
                {
                    grad_to_b_conv2_t[i] /= batch_size;
                }

                //L2 
                add_weight(grad_to_w1t, weight_to_first_hidden_layer, n_of_input_layer * n_of_first_hidden_layer);
                add_weight(grad_to_w4t, weight_to_output_layer, n_of_first_hidden_layer * n_of_output_layer);
                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    add_weight(grad_to_first_conv_filter_t[c].filter, first_conv_filter[c].filter, filter_hight * filter_width); 
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    add_weight(grad_to_second_conv_filter_t[c].filter, second_conv_filter[c].filter, filter_hight * filter_width); 
                }

                momentum_update(weight_to_first_hidden_layer, grad_to_w1t, velocity_grad_buffer_w1t, bias_of_first_hidden_layer, grad_to_b1t, velocity_grad_buffer_b1t, n_of_input_layer * n_of_first_hidden_layer, n_of_first_hidden_layer);
                momentum_update(weight_to_output_layer, grad_to_w4t, velocity_grad_buffer_w4t, bias_of_output_layer, grad_to_b4t, velocity_grad_buffer_b4t, n_of_first_hidden_layer * n_of_output_layer, n_of_output_layer);
                momentum_update_conv(first_conv_filter, grad_to_first_conv_filter_t, velocity_grad_buffer_f1t, first_conv_bias, grad_to_b_conv1_t, velocity_grad_buffer_conv_b1t, n_of_first_channel, n_of_first_channel);
                momentum_update_conv(second_conv_filter, grad_to_second_conv_filter_t, velocity_grad_buffer_f2t, second_conv_bias, grad_to_b_conv2_t, velocity_grad_buffer_conv_b2t, n_of_first_channel * n_of_second_channel, n_of_second_channel);

                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_first_conv_filter_t[c].filter[h * filter_width + w] = 0;
                        }
                        
                    }
                    
                }
                for (size_t i = 0; i < n_of_first_channel; i++)
                {
                    grad_to_b_conv1_t[i] = 0;
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_second_conv_filter_t[c].filter[h * filter_width + w] = 0;
                        }
                    
                    }
                
                }
                for (size_t i = 0; i < n_of_second_channel; i++)
                {
                    grad_to_b_conv2_t[i] = 0;
                }
            }

            free(training_image_buffer);
            free(training_label_buffer);
        }
        else {
            start = clock();
            int batch = 0;
            uint8_t *training_image_buffer = malloc(60000 * 784 * sizeof(uint8_t));
            uint8_t *training_label_buffer = malloc(60000 * sizeof(uint8_t));
            fseek(learning_data_images, 16, SEEK_SET);
            fread(training_image_buffer, sizeof(uint8_t), 60000 * 784, learning_data_images);
            fseek(learning_data_labels, 8, SEEK_SET);
            fread(training_label_buffer, sizeof(uint8_t), 60000, learning_data_labels);
        

            generate_dropout_mask(dropout_mask_for_first_hidden_layer, n_of_first_hidden_layer * (60000/batch_size));

            for (int loop = 0; loop < 60000; loop++){
            if (!(loop%1000) && debug) {printf("%d datas have been processed.\n", loop);}

            //inputting data
            for (int i = 0; i < 784; i++){
                input_image[i] = (float)(training_image_buffer[784 * order_indices[loop] + i])/255;
            }
            
            //inputting label
            answer = training_label_buffer[order_indices[loop]];

            //forward pass
            convolution_single_to_multi(input_image, first_conv_filter, first_conv_layer_pre_activation, 28, 28, n_of_first_channel);
            add_bias_conv(first_conv_layer_pre_activation, first_conv_bias, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1));
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                relu(first_conv_layer_pre_activation[i].layer, first_conv_layer_activation[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            maxpool(first_conv_layer_activation, first_maxpooling_layer, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1), 2);
            
            convolution_multi_to_multi(first_maxpooling_layer, second_conv_filter, second_conv_layer_pre_activation, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2, n_of_first_channel, n_of_second_channel);
            add_bias_conv(second_conv_layer_pre_activation, second_conv_bias, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1));
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                relu(second_conv_layer_pre_activation[i].layer, second_conv_layer_activation[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            maxpool(second_conv_layer_activation, second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1), 2);

            flatten(input_layer, second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_width + 1)/2 - filter_width + 1)/2);
            
            if (avx2 == true) {
                mat_vec_mul(weight_to_first_hidden_layer, input_layer, z1, n_of_first_hidden_layer, n_of_input_layer);
                vec_add_avx(z1, bias_of_first_hidden_layer, z1, n_of_first_hidden_layer);
            }
            else {
                mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
                add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
            }
            relu(z1, first_hidden_layer, n_of_first_hidden_layer);
            if (dropout == 1) {apply_dropout(first_hidden_layer, dropout_mask_for_first_hidden_layer, batch, n_of_first_hidden_layer);}

            if (avx2 == true) {
                mat_vec_mul(weight_to_output_layer, first_hidden_layer, zout, n_of_output_layer, n_of_first_hidden_layer);
                vec_add_avx(zout, bias_of_output_layer, zout, n_of_output_layer);
            }
            else {
                mmul(zout, first_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_first_hidden_layer);
                add_bias(zout, bias_of_output_layer, n_of_output_layer);
            }
            softmax(zout, output_layer, n_of_output_layer);

            //loss function (cross entropy)
            for (int i = 0; i < n_of_output_layer; i++){
                answer_arr[i] = 0.0;
            }
            answer_arr[answer] = 1.0;
            loss = 0.0f;
            for (int i = 0; i < n_of_output_layer; i++){
                loss += answer_arr[i] * logf(output_layer[i] + 1e-8f);
            }
            loss = -loss;
            //L2 regularization
            loss += (regularization_rate / 2.0f) *f_arr_squared_sum(weight_to_output_layer, n_of_output_layer * n_of_first_hidden_layer);
            loss += (regularization_rate / 2.0f) *f_arr_squared_sum(weight_to_first_hidden_layer, n_of_first_hidden_layer * n_of_input_layer);
            for (size_t c = 0; c < n_of_first_channel; c++)
            {
                loss += (regularization_rate / 2.0f) *f_arr_squared_sum(first_conv_filter[c].filter, filter_hight * filter_width);   
            }
            for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
            {
                loss += (regularization_rate / 2.0f) *f_arr_squared_sum(second_conv_filter[c].filter, filter_hight * filter_width);   
            }
            avg_loss += loss;
            
            //backward pass
            compute_output_delta(delta_4, output_layer, answer_arr, n_of_output_layer);

            weight_grad(delta_4, first_hidden_layer, grad_to_w4, n_of_output_layer, n_of_first_hidden_layer);
            grad_bias(delta_4, grad_to_b4, n_of_output_layer);

            compute_hidden_delta(delta_4, weight_to_output_layer, z1, delta_1, n_of_first_hidden_layer, n_of_output_layer);
            if (dropout == 1) {apply_dropout(delta_1, dropout_mask_for_first_hidden_layer, batch, n_of_first_hidden_layer);}

            weight_grad(delta_1, input_layer, grad_to_w1, n_of_first_hidden_layer, n_of_input_layer);
            grad_bias(delta_1, grad_to_b1, n_of_first_hidden_layer);

            compute_hidden_activation_delta(delta_1, weight_to_first_hidden_layer, delta_in, n_of_input_layer, n_of_first_hidden_layer);

            unflatten(backward_second_maxpool, delta_in, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_width + 1)/2 - filter_width + 1)/2);
            
            maxpool_backward(backward_second_maxpool, backward_second_conv, second_maxpooling_layer, n_of_second_channel, (28 - filter_hight + 1)/2 - filter_hight + 1, (28 - filter_width + 1)/2 - filter_width + 1, 2);
            
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                backward_relu(backward_second_conv[i].layer, backward_second_conv[i].layer, second_conv_layer_pre_activation[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            //バイアス勾配
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                grad_to_b_conv2[i] = float_array_sum(backward_second_conv[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            //フィルター勾配
            backward_conv_filter_multi_to_multi(grad_to_second_conv_filter, first_maxpooling_layer, backward_second_conv, n_of_second_channel, n_of_first_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
            

            //前層アクティベーション勾配
            backward_conv_layer(backward_first_maxpool, backward_second_conv, second_conv_filter, n_of_first_channel, n_of_second_channel, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2);
            
            maxpool_backward(backward_first_maxpool, backward_first_conv, first_maxpooling_layer, n_of_first_channel, 28 - filter_hight + 1, 28 - filter_width + 1, 2);
            
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                backward_relu(backward_first_conv[i].layer, backward_first_conv[i].layer, first_conv_layer_pre_activation[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            //バイアス勾配
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                grad_to_b_conv1[i] = float_array_sum(backward_first_conv[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            //フィルター勾配
            backward_conv_filter_single_to_multi(grad_to_first_conv_filter, input_image, backward_first_conv, n_of_first_channel, 28, 28);
            


            for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
            {
                grad_to_w1t[i] += grad_to_w1[i];
            }
            for (size_t i = 0; i < n_of_first_hidden_layer; i++)
            {
                grad_to_b1t[i] += grad_to_b1[i];
            }
            for (size_t i = 0; i < n_of_first_hidden_layer * n_of_output_layer; i++)
            {
                grad_to_w4t[i] += grad_to_w4[i];
            }
            for (size_t i = 0; i < n_of_output_layer; i++)
            {
                grad_to_b4t[i] += grad_to_b4[i];
            }
            for (size_t c = 0; c < n_of_first_channel; c++)
            {
                for (size_t h = 0; h < filter_hight; h++)
                {
                    for (size_t w = 0; w < filter_width; w++)
                    {
                        grad_to_first_conv_filter_t[c].filter[h * filter_width + w] += grad_to_first_conv_filter[c].filter[h * filter_width + w];
                    }
                    
                }
                
            }
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                grad_to_b_conv1_t[i] += grad_to_b_conv1[i];
            }
            for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
            {
                for (size_t h = 0; h < filter_hight; h++)
                {
                    for (size_t w = 0; w < filter_width; w++)
                    {
                        grad_to_second_conv_filter_t[c].filter[h * filter_width + w] += grad_to_second_conv_filter[c].filter[h * filter_width + w];
                    }
                    
                }
                
            }
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                grad_to_b_conv2_t[i] += grad_to_b_conv2[i];
            }

            //update params
            if (loop%batch_size == (batch_size-1)){
                //average
                for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
                {
                    grad_to_w1t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_first_hidden_layer; i++)
                {
                    grad_to_b1t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_first_hidden_layer * n_of_output_layer; i++)
                {
                    grad_to_w4t[i] /= batch_size;
                }
                for (size_t i = 0; i < n_of_output_layer; i++)
                {
                    grad_to_b4t[i] /= batch_size;
                }
                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_first_conv_filter_t[c].filter[h * filter_width + w] /= batch_size;
                        }
                        
                    }
                    
                }
                for (size_t i = 0; i < n_of_first_channel; i++)
                {
                    grad_to_b_conv1_t[i] /= batch_size;
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_second_conv_filter_t[c].filter[h * filter_width + w] /= batch_size;
                        }
                    
                    }
                
                }
                for (size_t i = 0; i < n_of_second_channel; i++)
                {
                    grad_to_b_conv2_t[i] /= batch_size;
                }

                //L2 
                add_weight(grad_to_w1t, weight_to_first_hidden_layer, n_of_input_layer * n_of_first_hidden_layer);
                add_weight(grad_to_w4t, weight_to_output_layer, n_of_first_hidden_layer * n_of_output_layer);
                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    add_weight(grad_to_first_conv_filter_t[c].filter, first_conv_filter[c].filter, filter_hight * filter_width); 
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    add_weight(grad_to_second_conv_filter_t[c].filter, second_conv_filter[c].filter, filter_hight * filter_width); 
                }

                momentum_update(weight_to_first_hidden_layer, grad_to_w1t, velocity_grad_buffer_w1t, bias_of_first_hidden_layer, grad_to_b1t, velocity_grad_buffer_b1t, n_of_input_layer * n_of_first_hidden_layer, n_of_first_hidden_layer);
                momentum_update(weight_to_output_layer, grad_to_w4t, velocity_grad_buffer_w4t, bias_of_output_layer, grad_to_b4t, velocity_grad_buffer_b4t, n_of_first_hidden_layer * n_of_output_layer, n_of_output_layer);
                momentum_update_conv(first_conv_filter, grad_to_first_conv_filter_t, velocity_grad_buffer_f1t, first_conv_bias, grad_to_b_conv1_t, velocity_grad_buffer_conv_b1t, n_of_first_channel, n_of_first_channel);
                momentum_update_conv(second_conv_filter, grad_to_second_conv_filter_t, velocity_grad_buffer_f2t, second_conv_bias, grad_to_b_conv2_t, velocity_grad_buffer_conv_b2t, n_of_first_channel * n_of_second_channel, n_of_second_channel);

                for (size_t c = 0; c < n_of_first_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_first_conv_filter_t[c].filter[h * filter_width + w] = 0;
                        }
                        
                    }
                    
                }
                for (size_t i = 0; i < n_of_first_channel; i++)
                {
                    grad_to_b_conv1_t[i] = 0;
                }
                for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
                {
                    for (size_t h = 0; h < filter_hight; h++)
                    {
                        for (size_t w = 0; w < filter_width; w++)
                        {
                            grad_to_second_conv_filter_t[c].filter[h * filter_width + w] = 0;
                        }
                    
                    }
                
                }
                for (size_t i = 0; i < n_of_second_channel; i++)
                {
                    grad_to_b_conv2_t[i] = 0;
                }

            }
        }
        end = clock();
        free(training_image_buffer);
        free(training_label_buffer);
        }
        printf("at epoch%d, training has finished. average loss:%f\n", epoch_loop+1, avg_loss / 60000);
        fprintf(fp, "epoch%d,%f,", epoch_loop+1, avg_loss / 60000);
        if (neck_check && !threaded) {printf("time: %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);}
        else if (neck_check && threaded) {printf("time: %f sec\n", (double)(end - start)/4 / CLOCKS_PER_SEC);}
        avg_loss = 0.0f;

            //testify section
        fseek(test_data_images, 16, SEEK_SET);
        fseek(test_data_labels, 8, SEEK_SET);
        for (int loop = 0; loop < 10000; loop++){
            if (!(loop%1000) && debug) {printf("%d datas have beed precessed.\n", loop);}
            
            //inputting data
            for (int i = 0; i < 784; i++)
            {
                input_image[i] = (float)(fgetc(test_data_images))/255;
            }

            //inputting label
            answer = fgetc(test_data_labels);

            //forward pass
            convolution_single_to_multi(input_image, first_conv_filter, first_conv_layer_pre_activation, 28, 28, n_of_first_channel);
            add_bias_conv(first_conv_layer_pre_activation, first_conv_bias, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1));
            for (size_t i = 0; i < n_of_first_channel; i++)
            {
                relu(first_conv_layer_pre_activation[i].layer, first_conv_layer_activation[i].layer, (28 - filter_hight + 1) * (28 - filter_width + 1));
            }

            maxpool(first_conv_layer_activation, first_maxpooling_layer, n_of_first_channel, (28 - filter_hight + 1), (28 - filter_width + 1), 2);
            
            convolution_multi_to_multi(first_maxpooling_layer, second_conv_filter, second_conv_layer_pre_activation, (28 - filter_hight + 1)/2, (28 - filter_width + 1)/2, n_of_first_channel, n_of_second_channel);
            add_bias_conv(second_conv_layer_pre_activation, second_conv_bias, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1));
            for (size_t i = 0; i < n_of_second_channel; i++)
            {
                relu(second_conv_layer_pre_activation[i].layer, second_conv_layer_activation[i].layer, ((28 - filter_hight + 1)/2 - filter_hight + 1) * ((28 - filter_width + 1)/2 - filter_width + 1));
            }

            maxpool(second_conv_layer_activation, second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1), ((28 - filter_width + 1)/2 - filter_width + 1), 2);

            flatten(input_layer, second_maxpooling_layer, n_of_second_channel, ((28 - filter_hight + 1)/2 - filter_hight + 1)/2, ((28 - filter_width + 1)/2 - filter_width + 1)/2);
            
            if (avx2 == true) {
                mat_vec_mul(weight_to_first_hidden_layer, input_layer, z1, n_of_first_hidden_layer, n_of_input_layer);
                vec_add_avx(z1, bias_of_first_hidden_layer, z1, n_of_first_hidden_layer);
            }
            else {
                mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
                add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
            }
            relu(z1, first_hidden_layer, n_of_first_hidden_layer);

            if (avx2 == true) {
                mat_vec_mul(weight_to_output_layer, first_hidden_layer, zout, n_of_output_layer, n_of_first_hidden_layer);
                vec_add_avx(zout, bias_of_output_layer, zout, n_of_output_layer);
            }
            else {
                mmul(zout, first_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_first_hidden_layer);
                add_bias(zout, bias_of_output_layer, n_of_output_layer);
            }
            softmax(zout, output_layer, n_of_output_layer);

            for (int i = 0; i < n_of_output_layer; i++)
            {
                answer_arr[i] = 0.0;
            }
            answer_arr[answer] = 1.0;
            loss = 0.0f;
            for (int i = 0; i < n_of_output_layer; i++)
            {
                loss += answer_arr[i] * logf(output_layer[i] + 1e-8f);
            }
            loss = -loss;
            //L2 regularization
            loss += (regularization_rate / 2.0f) *f_arr_squared_sum(weight_to_output_layer, n_of_output_layer * n_of_first_hidden_layer);
            loss += (regularization_rate / 2.0f) *f_arr_squared_sum(weight_to_first_hidden_layer, n_of_first_hidden_layer * n_of_input_layer);
            for (size_t c = 0; c < n_of_first_channel; c++)
            {
                loss += (regularization_rate / 2.0f) *f_arr_squared_sum(first_conv_filter[c].filter, filter_hight * filter_width);   
            }
            for (size_t c = 0; c < n_of_first_channel * n_of_second_channel; c++)
            {
                loss += (regularization_rate / 2.0f) *f_arr_squared_sum(second_conv_filter[c].filter, filter_hight * filter_width);   
            }
            avg_loss += loss;
            if (find_max_index(output_layer, n_of_output_layer) == answer)
            {
                ++hit;
            }
        }
        printf("at epoch%d, test has finished. average loss:%f, hit rate: %f%%\n", epoch_loop+1, avg_loss / 10000, (float)hit/100);
        fprintf(fp, "%f,%f\n", avg_loss / 10000, (float)hit/100);
    }

    //end
    free(input_image);
    free(input_layer);
    free(first_hidden_layer);
    free(output_layer);
    free(weight_to_first_hidden_layer);
    free(bias_of_first_hidden_layer);
    free(weight_to_output_layer);
    free(bias_of_output_layer);
    free(z1);
    free(zout);
    free(delta_4);
    free(delta_1);
    free(grad_to_b1);
    free(grad_to_b4);
    free(grad_to_w1);
    free(grad_to_w4);
    free(grad_to_b1t);
    free(grad_to_b4t);
    free(grad_to_w1t);
    free(grad_to_w4t);
    free(dropout_mask_for_first_hidden_layer);
    free(delta_in);
    input_layer = NULL;
    first_hidden_layer = NULL;
    output_layer = NULL;
    weight_to_first_hidden_layer = NULL;
    bias_of_first_hidden_layer = NULL;
    weight_to_output_layer = NULL;
    bias_of_output_layer = NULL;
    z1 = NULL;
    zout = NULL;
    delta_4 = NULL;
    delta_1 = NULL;
    grad_to_b1 = NULL;
    grad_to_b4 = NULL;
    grad_to_w1 = NULL;
    grad_to_w4 = NULL;
    grad_to_b1t = NULL;
    grad_to_b4t = NULL;
    grad_to_w1t = NULL;
    grad_to_w4t = NULL;
    //free_workspace(ws, 4);
    free_conv_layer(first_conv_layer_pre_activation, n_of_first_channel);
    free_conv_layer(first_conv_layer_activation, n_of_first_channel);
    free_conv_layer(second_conv_layer_pre_activation, n_of_second_channel);
    free_conv_layer(second_conv_layer_activation, n_of_second_channel);
    free_filter(first_conv_filter, n_of_first_channel);
    free_filter(second_conv_filter, n_of_first_channel * n_of_second_channel);
    free_maxpool_layer(first_maxpooling_layer, n_of_first_channel);
    free_maxpool_layer(second_maxpooling_layer, n_of_second_channel);
    free_maxpool_layer(backward_second_maxpool, n_of_second_channel);
    free_conv_layer(backward_second_conv, n_of_second_channel);
    free_maxpool_layer(backward_first_maxpool, n_of_first_channel);
    free_conv_layer(backward_first_conv, n_of_first_channel);
    free_filter(grad_to_first_conv_filter, n_of_first_channel);
    free_filter(grad_to_second_conv_filter, n_of_first_channel * n_of_second_channel);
    free(first_conv_bias);
    free(second_conv_bias);
    free_filter(grad_to_first_conv_filter_t, n_of_first_channel);
    free_filter(grad_to_second_conv_filter_t, n_of_first_channel * n_of_second_channel);
    free(grad_to_b_conv1_t);
    free(grad_to_b_conv2_t);
    free(order_indices);
    fclose(learning_data_images);
    fclose(learning_data_labels);
    fclose(test_data_images);
    fclose(test_data_labels);
    fclose(fp);

    //free momentum buffer
    free(velocity_grad_buffer_w1t);
    free(velocity_grad_buffer_w4t);
    free(velocity_grad_buffer_b1t);
    free(velocity_grad_buffer_b4t);
    free_filter(velocity_grad_buffer_f1t, n_of_first_channel);
    free_filter(velocity_grad_buffer_f2t, n_of_first_channel * n_of_second_channel);
    free(velocity_grad_buffer_conv_b1t);
    free(velocity_grad_buffer_conv_b2t);

    //free workspace
    free_workspace(ws, num_threads);

    //destroy mutex
    pthread_mutex_destroy(&mutex);
    return 0;
}