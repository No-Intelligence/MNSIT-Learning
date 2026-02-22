#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>

#define PI 3.14159265358979
#define n_of_input_layer 784
#define n_of_first_hidden_layer 1024
#define n_of_second_hidden_layer 512
#define n_of_third_hidden_layer 256
#define n_of_output_layer 10
#define learning_rate 0.001
#define batch_size 96
#define epoch 20
#define debug 0
#define neck_check 0
#define threaded 0
#define train_images "train-images-fashion.idx3-ubyte"
#define train_labels "train-labels-fashion.idx1-ubyte"
#define test_images "t10k-images-fashion.idx3-ubyte"
#define test_labels "t10k-labels-fashion.idx1-ubyte"

typedef struct {
    //buffer
    uint8_t *training_image_buffer, *training_label_buffer;

    //weights
    float  *w1, *w2, *w3, *wout, *b1, *b2, *b3, *bout;

    //activations
    float *a_in, *a_1, *a_2, *a_3, *a_out;
    float *z_1, *z_2, *z_3, *z_out;

    //deltas
    float *delta_1, *delta_2, *delta_3, *delta_4;

    //gradient
    float *grad_w1, *grad_w2, *grad_w3, *grad_w4, *grad_b1, *grad_b2, *grad_b3, *grad_b4;

    //gradient total
    float *grad_w1t, *grad_w2t, *grad_w3t, *grad_w4t, *grad_b1t, *grad_b2t, *grad_b3t, *grad_b4t;
} thread_workspace_t;

thread_workspace_t* alloc_workspace (int n_threads) {
    thread_workspace_t *p = calloc(n_threads, sizeof(thread_workspace_t));
    for (size_t i = 0; i < n_threads; i++)
    {
        p[i].training_image_buffer = malloc(784 * batch_size / 4 * sizeof(uint8_t));
        p[i].training_label_buffer = malloc(batch_size / 4 * sizeof(uint8_t));
        /*p[i].w1 = malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
        p[i].w2 = malloc(n_of_first_hidden_layer * n_of_second_hidden_layer * sizeof(float));
        p[i].w3 = malloc(n_of_second_hidden_layer * n_of_third_hidden_layer * sizeof(float));
        p[i].wout = malloc(n_of_third_hidden_layer * n_of_output_layer * sizeof(float));
        p[i].b1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].b2 = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].b3 = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].bout = malloc(n_of_output_layer * sizeof(float));*/
        p[i].a_in = malloc(n_of_input_layer * sizeof(float));
        p[i].a_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].a_2 = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].a_3 = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].a_out = malloc(n_of_output_layer * sizeof(float));
        p[i].z_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].z_2 = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].z_3 = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].z_out = malloc(n_of_output_layer * sizeof(float));
        p[i].delta_1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].delta_2 = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].delta_3 = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].delta_4 = malloc(n_of_output_layer * sizeof(float));
        p[i].grad_w1 = malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
        p[i].grad_w2 = malloc(n_of_first_hidden_layer * n_of_second_hidden_layer * sizeof(float));
        p[i].grad_w3 = malloc(n_of_second_hidden_layer * n_of_third_hidden_layer * sizeof(float));
        p[i].grad_w4 = malloc(n_of_third_hidden_layer * n_of_output_layer * sizeof(float));
        p[i].grad_b1 = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].grad_b2 = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].grad_b3 = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].grad_b4 = malloc(n_of_output_layer * sizeof(float));
        p[i].grad_w1t = malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
        p[i].grad_w2t = malloc(n_of_first_hidden_layer * n_of_second_hidden_layer * sizeof(float));
        p[i].grad_w3t = malloc(n_of_second_hidden_layer * n_of_third_hidden_layer * sizeof(float));
        p[i].grad_w4t = malloc(n_of_third_hidden_layer * n_of_output_layer * sizeof(float));
        p[i].grad_b1t = malloc(n_of_first_hidden_layer * sizeof(float));
        p[i].grad_b2t = malloc(n_of_second_hidden_layer * sizeof(float));
        p[i].grad_b3t = malloc(n_of_third_hidden_layer * sizeof(float));
        p[i].grad_b4t = malloc(n_of_output_layer * sizeof(float));
    }
    return p;
}

void free_workspace (thread_workspace_t *p, int n_threads) {
    for (size_t i = 0; i < n_threads; i++)
    {
        //buffer
        free(p[i].training_image_buffer); free(p[i].training_label_buffer);

        //weights
        /*free(p[i].w1); free(p[i].w2); free(p[i].w3); free(p[i].wout); free(p[i].b1); free(p[i].b2); free(p[i].b3); free(p[i].bout);*/

        //activations
        free(p[i].a_in); free(p[i].a_1); free(p[i].a_2); free(p[i].a_3); free(p[i].a_out);
        free(p[i].z_1); free(p[i].z_2); free(p[i].z_3); free(p[i].z_out);
        
        //deltas
        free(p[i].delta_1); free(p[i].delta_2); free(p[i].delta_3); free(p[i].delta_4);
        
        //gradients
        free(p[i].grad_w1); free(p[i].grad_w2); free(p[i].grad_w3); free(p[i].grad_w4); free(p[i].grad_b1); free(p[i].grad_b2); free(p[i].grad_b3); free(p[i].grad_b4);
        
        //gradients total
        free(p[i].grad_w1t); free(p[i].grad_w2t); free(p[i].grad_w3t); free(p[i].grad_w4t); free(p[i].grad_b1t); free(p[i].grad_b2t); free(p[i].grad_b3t); free(p[i].grad_b4t);
    }
    free(p);
}

void add_four_arr(float *output, float *input1, float *input2, float *input3, float *input4, int n_of_arr){
    for (size_t i = 0; i < n_of_arr; i++)
    {
        output[i] = (input1[i] + input2[i] + input3[i] + input4[i]) / batch_size;
    }
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

int mmul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr){
    for (int i = 0; i < n_of_output_arr; i++)
    {
        output_arr[i] = 0.0;
    }
    
    for (int i = 0; i < n_of_output_arr; i++)
    {
        for (int j = 0; j < n_of_input_arr; j++)
        {
            output_arr[i] += matrix[n_of_input_arr * i + j] * input_arr[j];
        }
        
    }

    return 0;
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
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        input_arr[i] = exp(input_arr[i] - max);
        sum += input_arr[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = input_arr[j] / sum;
    }    
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
            output[i] = 0.0f;
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

void update_params(float *weight, float *weight_grad, float *bias, float *bias_grad, int n_of_weight, int n_of_bias){
    for (int i = 0; i < n_of_weight; i++)
    {
        weight[i] += -1 * learning_rate * weight_grad[i];
        weight_grad[i] = 0.0f;
    }
    for (int i = 0; i < n_of_bias; i++)
    {
        bias[i] += -1 * learning_rate * bias_grad[i];
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

void* training_threaded (void* arg){
    //standby section
    thread_workspace_t* info = (thread_workspace_t*)arg;
    int answer;
    float answer_arr[10];
    float loss;
    
    //compute section
        for (int loop = 0; loop < batch_size / 4; loop++){

            //inputting data
            for (int i = 0; i < n_of_input_layer; i++){
                info->a_in[i] = (float)(info->training_image_buffer[784 * loop + i])/255;
            }

            //inputting label
            answer = info->training_label_buffer[loop];

            //forward pass
            mmul(info->z_1, info->a_in, info->w1, n_of_first_hidden_layer, n_of_input_layer);
            add_bias(info->z_1, info->b1, n_of_first_hidden_layer);
            relu(info->z_1, info->a_1, n_of_first_hidden_layer);

            mmul(info->z_2, info->a_1, info->w2, n_of_second_hidden_layer, n_of_first_hidden_layer);
            add_bias(info->z_2, info->b2, n_of_second_hidden_layer);
            relu(info->z_2, info->a_2, n_of_second_hidden_layer);

            mmul(info->z_3, info->a_2, info->w3, n_of_third_hidden_layer, n_of_second_hidden_layer);
            add_bias(info->z_3, info->b3, n_of_third_hidden_layer);
            relu(info->z_3, info->a_3, n_of_third_hidden_layer);

            mmul(info->z_out, info->a_3, info->wout, n_of_output_layer, n_of_third_hidden_layer);
            add_bias(info->z_out, info->bout, n_of_output_layer);
            softmax(info->z_out, info->a_out, n_of_output_layer);

            //loss function (cross entropy)
            for (int i = 0; i < n_of_output_layer; i++){
                answer_arr[i] = 0.0;
            }
            answer_arr[answer] = 1.0;
            loss = 0.0f;
            for (int i = 0; i < n_of_output_layer; i++){
                loss += answer_arr[i] * logf(info->a_out[i] + 1e-8f);
            }
            loss = -loss;

            //backward pass
            compute_output_delta(info->delta_4, info->a_out, answer_arr, n_of_output_layer);

            weight_grad(info->delta_4, info->a_3, info->grad_w4, n_of_output_layer, n_of_third_hidden_layer);
            grad_bias(info->delta_4, info->grad_b4, n_of_output_layer);

            compute_hidden_delta(info->delta_4, info->wout, info->z_3, info->delta_3, n_of_third_hidden_layer, n_of_output_layer);

            weight_grad(info->delta_3, info->a_2, info->grad_w3, n_of_third_hidden_layer, n_of_second_hidden_layer);
            grad_bias(info->delta_3, info->grad_b3, n_of_third_hidden_layer);

            compute_hidden_delta(info->delta_3, info->w3, info->z_2, info->delta_2, n_of_second_hidden_layer, n_of_third_hidden_layer);

            weight_grad(info->delta_2, info->a_1, info->grad_w2, n_of_second_hidden_layer, n_of_first_hidden_layer);
            grad_bias(info->delta_2, info->grad_b2, n_of_second_hidden_layer);

            compute_hidden_delta(info->delta_2, info->w2, info->z_1, info->delta_1, n_of_first_hidden_layer, n_of_second_hidden_layer);

            weight_grad(info->delta_1, info->a_in, info->grad_w1, n_of_first_hidden_layer, n_of_input_layer);
            grad_bias(info->delta_1, info->grad_b1, n_of_first_hidden_layer);

            for (int i = 0; i < n_of_first_hidden_layer; i++){
                info->grad_b1t[i] += (float)info->grad_b1[i];
            }
            for (int i = 0; i < n_of_second_hidden_layer; i++){
                info->grad_b2t[i] += (float)info->grad_b2[i];
            }
            for (int i = 0; i < n_of_third_hidden_layer; i++){
                info->grad_b3t[i] += (float)info->grad_b3[i];
            }
            for (int i = 0; i < n_of_output_layer; i++){
                info->grad_b4t[i] += (float)info->grad_b4[i];
            }
            for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++){
                info->grad_w1t[i] += (float)info->grad_w1[i];
            }
            for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++){
                info->grad_w2t[i] += (float)info->grad_w2[i];
            }
            for (int i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++){
                info->grad_w3t[i] += (float)info->grad_w3[i];
            }
            for (int i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++){
                info->grad_w4t[i] += (float)info->grad_w4[i];
            }
        }

        return NULL;
}

int main (void){
    //define variables
    int answer = 0, hit = 0;
    float answer_arr[n_of_output_layer] = {0.0f};
    float loss = 0.0f, avg_loss = 0.0f;
    int order_indices[60000];
    for (int i = 0; i < 60000; i++)
    {
        order_indices[i] = i;
    }
    clock_t start, end;
    pthread_t th1, th2, th3, th4;
    
    
    //define pointer
    float *input_layer;
    float *first_hidden_layer;
    float *second_hidden_layer;
    float *third_hidden_layer;
    float *output_layer;
    float *weight_to_first_hidden_layer;
    float *bias_of_first_hidden_layer;
    float *weight_to_second_hidden_layer;
    float *bias_of_second_hidden_layer;
    float *weight_to_third_hidden_layer;
    float *bias_of_third_hidden_layer;
    float *weight_to_output_layer;
    float *bias_of_output_layer;
    float *z1;
    float *z2;
    float *z3;
    float *zout;
    float *delta_4, *delta_3, *delta_2, *delta_1;
    float *grad_to_w4, *grad_to_b4, *grad_to_w3, *grad_to_b3, *grad_to_w2, *grad_to_b2, *grad_to_w1, *grad_to_b1;
    float *grad_to_w4t, *grad_to_b4t, *grad_to_w3t, *grad_to_b3t, *grad_to_w2t, *grad_to_b2t, *grad_to_w1t, *grad_to_b1t;

    //allocetion params
    input_layer = (float*)malloc(n_of_input_layer * sizeof(float));
    first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    weight_to_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * n_of_input_layer * sizeof(float));
    bias_of_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    weight_to_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * n_of_first_hidden_layer * sizeof(float));
    bias_of_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    weight_to_third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * n_of_second_hidden_layer * sizeof(float));
    bias_of_third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    weight_to_output_layer = (float*)malloc(n_of_output_layer * n_of_third_hidden_layer * sizeof(float));
    bias_of_output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    z1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    z2 = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    z3 = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    zout = (float*)malloc(n_of_output_layer * sizeof(float));
    delta_4 = (float*)malloc(n_of_output_layer * sizeof(float));
    delta_3 = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    delta_2 = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    delta_1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    grad_to_w4 = (float*)malloc(n_of_third_hidden_layer * n_of_output_layer * sizeof(float));
    grad_to_w3 = (float*)malloc(n_of_second_hidden_layer * n_of_third_hidden_layer * sizeof(float));
    grad_to_w2 = (float*)malloc(n_of_first_hidden_layer * n_of_second_hidden_layer * sizeof(float));
    grad_to_w1 = (float*)malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
    grad_to_b4 = (float*)malloc(n_of_output_layer * sizeof(float));
    grad_to_b3 = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    grad_to_b2 = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    grad_to_b1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    grad_to_w4t = (float*)malloc(n_of_third_hidden_layer * n_of_output_layer * sizeof(float));
    grad_to_w3t = (float*)malloc(n_of_second_hidden_layer * n_of_third_hidden_layer * sizeof(float));
    grad_to_w2t = (float*)malloc(n_of_first_hidden_layer * n_of_second_hidden_layer * sizeof(float));
    grad_to_w1t = (float*)malloc(n_of_input_layer * n_of_first_hidden_layer * sizeof(float));
    grad_to_b4t = (float*)malloc(n_of_output_layer * sizeof(float));
    grad_to_b3t = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    grad_to_b2t = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    grad_to_b1t = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    thread_workspace_t *ws = alloc_workspace(4);

    //file
    FILE *learning_data_images, *learning_data_labels, *test_data_images, *test_data_labels, *fp;

    //weight initialize
    srand(time(NULL));
    for (int i = 0; i < n_of_first_hidden_layer; i++)
    {
        bias_of_first_hidden_layer[i] = 0;
    }
    for (int i = 0; i < n_of_second_hidden_layer; i++)
    {
        bias_of_second_hidden_layer[i] = 0;
    }
    for (int i = 0; i < n_of_third_hidden_layer; i++)
    {
        bias_of_third_hidden_layer[i] = 0;
    }
    for (int i = 0; i < n_of_output_layer; i++)
    {
        bias_of_output_layer[i] = 0;
    }
    he_initialize_uniform(weight_to_first_hidden_layer, n_of_input_layer, n_of_first_hidden_layer);
    he_initialize_uniform(weight_to_second_hidden_layer, n_of_first_hidden_layer, n_of_second_hidden_layer);
    he_initialize_uniform(weight_to_third_hidden_layer, n_of_second_hidden_layer, n_of_third_hidden_layer);
    he_initialize_uniform(weight_to_output_layer, n_of_third_hidden_layer, n_of_output_layer);
    for (int i = 0; i < n_of_first_hidden_layer; i++)
    {
        grad_to_b1t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_second_hidden_layer; i++)
    {
        grad_to_b2t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_third_hidden_layer; i++)
    {
        grad_to_b3t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_output_layer; i++)
    {
        grad_to_b4t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
    {
        grad_to_w1t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++)
    {
        grad_to_w2t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++)
    {
        grad_to_w3t[i] = 0.0f;
    }
    for (int i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++)
    {
        grad_to_w4t[i] = 0.0f;
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


    printf("Training start\n");
    for (int epoch_loop = 0; epoch_loop < epoch; epoch_loop++){
        avg_loss = 0.0f;
        hit = 0;
        shuffle_indices(order_indices, 60000);

        //learning section
        if (threaded == 1)
        {
            start = clock();
            //standby section
            for (size_t i = 0; i < 4; i++)
            {
                ws[i].w1 = weight_to_first_hidden_layer;
                ws[i].w2 = weight_to_second_hidden_layer;
                ws[i].w3 = weight_to_third_hidden_layer;
                ws[i].wout = weight_to_output_layer;
                ws[i].b1 = bias_of_first_hidden_layer;
                ws[i].b2 = bias_of_second_hidden_layer;
                ws[i].b3 = bias_of_third_hidden_layer;
                ws[i].bout = bias_of_output_layer;
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
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_input_layer * n_of_first_hidden_layer; j++)
                    {
                        ws[i].grad_w1t[j] = 0.0f;
                    }

                }  
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_first_hidden_layer * n_of_second_hidden_layer; j++)
                    {
                        ws[i].grad_w2t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_second_hidden_layer * n_of_third_hidden_layer; j++)
                    {
                        ws[i].grad_w3t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_third_hidden_layer * n_of_output_layer; j++)
                    {
                        ws[i].grad_w4t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_first_hidden_layer; j++)
                    {
                        ws[i].grad_b1t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_second_hidden_layer; j++)
                    {
                        ws[i].grad_b2t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_third_hidden_layer; j++)
                    {
                        ws[i].grad_b3t[j] = 0.0f;
                    }
                
                }
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < n_of_output_layer; j++)
                    {
                        ws[i].grad_b4t[j] = 0.0f;
                    }
                
                }

                //datas inport
                for (size_t i = 0; i < 4; i++)
                {
                    for (size_t j = 0; j < batch_size / 4; j++)
                    {
                        for (size_t copy_image = 0; copy_image < 784; copy_image++)
                        {
                            ws[i].training_image_buffer[784 * j + copy_image] = training_image_buffer[784 * order_indices[15000 * i + j + batch_size / 4 * loop] + copy_image];
                        }
                        
                        ws[i].training_label_buffer[j] = training_label_buffer[order_indices[15000 * i + j + batch_size / 4 * loop]];
                    }
                    
                }
                
                //compute grad
                pthread_create(&th1, NULL, training_threaded, &ws[0]);
                pthread_create(&th2, NULL, training_threaded, &ws[1]);
                pthread_create(&th3, NULL, training_threaded, &ws[2]);
                pthread_create(&th4, NULL, training_threaded, &ws[3]);
                pthread_join(th1, NULL);
                pthread_join(th2, NULL);
                pthread_join(th3, NULL);
                pthread_join(th4, NULL);

                //update params
                add_four_arr(grad_to_b1t, ws[0].grad_b1t, ws[1].grad_b1t, ws[2].grad_b1t, ws[3].grad_b1t, n_of_first_hidden_layer);
                add_four_arr(grad_to_b2t, ws[0].grad_b2t, ws[1].grad_b2t, ws[2].grad_b2t, ws[3].grad_b2t, n_of_second_hidden_layer);
                add_four_arr(grad_to_b3t, ws[0].grad_b3t, ws[1].grad_b3t, ws[2].grad_b3t, ws[3].grad_b3t, n_of_third_hidden_layer);
                add_four_arr(grad_to_b4t, ws[0].grad_b4t, ws[1].grad_b4t, ws[2].grad_b4t, ws[3].grad_b4t, n_of_output_layer);
                add_four_arr(grad_to_w1t, ws[0].grad_w1t, ws[1].grad_w1t, ws[2].grad_w1t, ws[3].grad_w1t, n_of_first_hidden_layer * n_of_input_layer);
                add_four_arr(grad_to_w2t, ws[0].grad_w2t, ws[1].grad_w2t, ws[2].grad_w2t, ws[3].grad_w2t, n_of_second_hidden_layer * n_of_first_hidden_layer);
                add_four_arr(grad_to_w3t, ws[0].grad_w3t, ws[1].grad_w3t, ws[2].grad_w3t, ws[3].grad_w3t, n_of_third_hidden_layer * n_of_second_hidden_layer);
                add_four_arr(grad_to_w4t, ws[0].grad_w4t, ws[1].grad_w4t, ws[2].grad_w4t, ws[3].grad_w4t, n_of_output_layer * n_of_third_hidden_layer);

                update_params(weight_to_first_hidden_layer, grad_to_w1t, bias_of_first_hidden_layer, grad_to_b1t, n_of_input_layer * n_of_first_hidden_layer, n_of_first_hidden_layer);
                update_params(weight_to_second_hidden_layer, grad_to_w2t, bias_of_second_hidden_layer, grad_to_b2t, n_of_first_hidden_layer * n_of_second_hidden_layer, n_of_second_hidden_layer);
                update_params(weight_to_third_hidden_layer, grad_to_w3t, bias_of_third_hidden_layer, grad_to_b3t, n_of_second_hidden_layer * n_of_third_hidden_layer, n_of_third_hidden_layer);
                update_params(weight_to_output_layer, grad_to_w4t, bias_of_output_layer, grad_to_b4t, n_of_third_hidden_layer * n_of_output_layer, n_of_output_layer);
            }

            free(training_image_buffer);
            free(training_label_buffer);
            end = clock();
        }
        else {
            start = clock();
        for (int loop = 0; loop < 60000; loop++){
            if (!(loop%1000) && debug) {printf("%d datas have been processed.\n", loop);}

            //offset data
            fseek(learning_data_images, 16 + 784 * order_indices[loop], SEEK_SET);
            fseek(learning_data_labels, 8 + order_indices[loop], SEEK_SET);

            //inputting data
            for (int i = 0; i < n_of_input_layer; i++){
                input_layer[i] = (float)(fgetc(learning_data_images))/255;
            }
            
            //inputting label
            answer = fgetc(learning_data_labels);

            //forward pass
            mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
            add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
            relu(z1, first_hidden_layer, n_of_first_hidden_layer);

            mmul(z2, first_hidden_layer, weight_to_second_hidden_layer, n_of_second_hidden_layer, n_of_first_hidden_layer);
            add_bias(z2, bias_of_second_hidden_layer, n_of_second_hidden_layer);
            relu(z2, second_hidden_layer, n_of_second_hidden_layer);

            mmul(z3, second_hidden_layer, weight_to_third_hidden_layer, n_of_third_hidden_layer, n_of_second_hidden_layer);
            add_bias(z3, bias_of_third_hidden_layer, n_of_third_hidden_layer);
            relu(z3, third_hidden_layer, n_of_third_hidden_layer);

            mmul(zout, third_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_third_hidden_layer);
            add_bias(zout, bias_of_output_layer, n_of_output_layer);
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
            avg_loss += loss;

            //backward pass
            compute_output_delta(delta_4, output_layer, answer_arr, n_of_output_layer);

            weight_grad(delta_4, third_hidden_layer, grad_to_w4, n_of_output_layer, n_of_third_hidden_layer);
            grad_bias(delta_4, grad_to_b4, n_of_output_layer);

            compute_hidden_delta(delta_4, weight_to_output_layer, z3, delta_3, n_of_third_hidden_layer, n_of_output_layer);

            weight_grad(delta_3, second_hidden_layer, grad_to_w3, n_of_third_hidden_layer, n_of_second_hidden_layer);
            grad_bias(delta_3, grad_to_b3, n_of_third_hidden_layer);

            compute_hidden_delta(delta_3, weight_to_third_hidden_layer, z2, delta_2, n_of_second_hidden_layer, n_of_third_hidden_layer);

            weight_grad(delta_2, first_hidden_layer, grad_to_w2, n_of_second_hidden_layer, n_of_first_hidden_layer);
            grad_bias(delta_2, grad_to_b2, n_of_second_hidden_layer);

            compute_hidden_delta(delta_2, weight_to_second_hidden_layer, z1, delta_1, n_of_first_hidden_layer, n_of_second_hidden_layer);

            weight_grad(delta_1, input_layer, grad_to_w1, n_of_first_hidden_layer, n_of_input_layer);
            grad_bias(delta_1, grad_to_b1, n_of_first_hidden_layer);

            for (int i = 0; i < n_of_first_hidden_layer; i++){
                grad_to_b1t[i] += (float)grad_to_b1[i]/batch_size;
            }
            for (int i = 0; i < n_of_second_hidden_layer; i++){
                grad_to_b2t[i] += (float)grad_to_b2[i]/batch_size;
            }
            for (int i = 0; i < n_of_third_hidden_layer; i++){
                grad_to_b3t[i] += (float)grad_to_b3[i]/batch_size;
            }
            for (int i = 0; i < n_of_output_layer; i++){
                grad_to_b4t[i] += (float)grad_to_b4[i]/batch_size;
            }
            for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++){
                grad_to_w1t[i] += (float)grad_to_w1[i]/batch_size;
            }
            for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++){
                grad_to_w2t[i] += (float)grad_to_w2[i]/batch_size;
            }
            for (int i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++){
                grad_to_w3t[i] += (float)grad_to_w3[i]/batch_size;
            }
            for (int i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++){
                grad_to_w4t[i] += (float)grad_to_w4[i]/batch_size;
            }

            //update params
            if (loop%batch_size == (batch_size-1)){
                update_params(weight_to_first_hidden_layer, grad_to_w1t, bias_of_first_hidden_layer, grad_to_b1t, n_of_input_layer * n_of_first_hidden_layer, n_of_first_hidden_layer);
                update_params(weight_to_second_hidden_layer, grad_to_w2t, bias_of_second_hidden_layer, grad_to_b2t, n_of_first_hidden_layer * n_of_second_hidden_layer, n_of_second_hidden_layer);
                update_params(weight_to_third_hidden_layer, grad_to_w3t, bias_of_third_hidden_layer, grad_to_b3t, n_of_second_hidden_layer * n_of_third_hidden_layer, n_of_third_hidden_layer);
                update_params(weight_to_output_layer, grad_to_w4t, bias_of_output_layer, grad_to_b4t, n_of_third_hidden_layer * n_of_output_layer, n_of_output_layer);
                for (int i = 0; i < n_of_first_hidden_layer; i++){
                    grad_to_b1t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_second_hidden_layer; i++){
                    grad_to_b2t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_third_hidden_layer; i++){
                    grad_to_b3t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_output_layer; i++){
                    grad_to_b4t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++){
                    grad_to_w1t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++){
                    grad_to_w2t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++){
                    grad_to_w3t[i] = 0.0f;
                }
                for (int i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++){
                    grad_to_w4t[i] = 0.0f;
                }
            }
        }
        end = clock();
        }
        printf("at epoch%d, training has finished. average loss:%f\n", epoch_loop+1, avg_loss / 60000);
        fprintf(fp, "epoch%d,%f,", epoch_loop+1, avg_loss / 60000);
        if (neck_check && !threaded) {printf("time: %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);}
        else if (neck_check && threaded) {printf("time: %f sec\n", (double)(end - start)/4 / CLOCKS_PER_SEC);}
        fseek(learning_data_images, -784 * 60000, SEEK_CUR);
        fseek(learning_data_labels, -60000, SEEK_CUR);
        avg_loss = 0.0f;

        //testify section
        for (int loop = 0; loop < 10000; loop++){
            if (!(loop%1000) && debug) {printf("%d datas have beed precessed.\n", loop);}
            
            //inputting data
            for (int i = 0; i < n_of_input_layer; i++)
            {
                input_layer[i] = (float)(fgetc(test_data_images))/255;
            }

            //inputting label
            answer = fgetc(test_data_labels);

            //forward pass
            mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
            add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
            relu(z1, first_hidden_layer, n_of_first_hidden_layer);

            mmul(z2, first_hidden_layer, weight_to_second_hidden_layer, n_of_second_hidden_layer, n_of_first_hidden_layer);
            add_bias(z2, bias_of_second_hidden_layer, n_of_second_hidden_layer);
            relu(z2, second_hidden_layer, n_of_second_hidden_layer);

            mmul(z3, second_hidden_layer, weight_to_third_hidden_layer, n_of_third_hidden_layer, n_of_second_hidden_layer);
            add_bias(z3, bias_of_third_hidden_layer, n_of_third_hidden_layer);
            relu(z3, third_hidden_layer, n_of_third_hidden_layer);

            mmul(zout, third_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_third_hidden_layer);
            add_bias(zout, bias_of_output_layer, n_of_output_layer);
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
            avg_loss += loss;
            if (find_max_index(output_layer, n_of_output_layer) == answer)
            {
                ++hit;
            }
        }
        printf("at epoch%d, test has finished. average loss:%f, hit rate: %f%%\n", epoch_loop+1, avg_loss / 10000, (float)hit/100);
        fprintf(fp, "%f,%f\n", avg_loss / 10000, (float)hit/100);
        fseek(test_data_images, -784 * 10000, SEEK_CUR);
        fseek(test_data_labels, -10000, SEEK_CUR);
    }


    //end
    free(input_layer);
    free(first_hidden_layer);
    free(second_hidden_layer);
    free(third_hidden_layer);
    free(output_layer);
    free(weight_to_first_hidden_layer);
    free(bias_of_first_hidden_layer);
    free(weight_to_second_hidden_layer);
    free(bias_of_second_hidden_layer);
    free(weight_to_third_hidden_layer);
    free(bias_of_third_hidden_layer);
    free(weight_to_output_layer);
    free(bias_of_output_layer);
    free(z1);
    free(z2);
    free(z3);
    free(zout);
    free(delta_4);
    free(delta_3);
    free(delta_2);
    free(delta_1);
    free(grad_to_b1);
    free(grad_to_b2);
    free(grad_to_b3);
    free(grad_to_b4);
    free(grad_to_w1);
    free(grad_to_w2);
    free(grad_to_w3);
    free(grad_to_w4);
    free(grad_to_b1t);
    free(grad_to_b2t);
    free(grad_to_b3t);
    free(grad_to_b4t);
    free(grad_to_w1t);
    free(grad_to_w2t);
    free(grad_to_w3t);
    free(grad_to_w4t);
    input_layer = NULL;
    first_hidden_layer = NULL;
    second_hidden_layer = NULL;
    output_layer = NULL;
    weight_to_first_hidden_layer = NULL;
    bias_of_first_hidden_layer = NULL;
    weight_to_second_hidden_layer = NULL;
    bias_of_second_hidden_layer = NULL;
    weight_to_output_layer = NULL;
    bias_of_output_layer = NULL;
    z1 = NULL;
    z2 = NULL;
    zout = NULL;
    delta_4 = NULL;
    delta_3 = NULL;
    delta_2 = NULL;
    delta_1 = NULL;
    grad_to_b1 = NULL;
    grad_to_b2 = NULL;
    grad_to_b3 = NULL;
    grad_to_b4 = NULL;
    grad_to_w1 = NULL;
    grad_to_w2 = NULL;
    grad_to_w3 = NULL;
    grad_to_b4 = NULL;
    grad_to_b1t = NULL;
    grad_to_b2t = NULL;
    grad_to_b3t = NULL;
    grad_to_b4t = NULL;
    grad_to_w1t = NULL;
    grad_to_w2t = NULL;
    grad_to_w3t = NULL;
    grad_to_b4t = NULL;
    free_workspace(ws, 4);
    fclose(learning_data_images);
    fclose(learning_data_labels);
    fclose(test_data_images);
    fclose(test_data_labels);
    fclose(fp);
    return 0;
}