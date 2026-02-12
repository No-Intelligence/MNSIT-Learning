#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979
#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_third_hidden_layer 128
#define n_of_output_layer 10
#define learning_rate 0.0015
#define batch_size 32
#define epoch 10
#define debug 1
#define neck_check 0
#define train_images "train-images.idx3-ubyte"
#define train_labels "train-labels.idx1-ubyte"
#define test_images "t10k-images.idx3-ubyte" 
#define test_labels "t10k-labels.idx1-ubyte"

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

    //file
    FILE *learning_data_images, *learning_data_labels, *test_data_images, *test_data_labels;

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

    //offset data
    fseek(test_data_images, 16, 0);
    fseek(test_data_labels, 8, 0);


    printf("learning start\n");
    for (int epoch_loop = 0; epoch_loop < epoch; epoch_loop++){
        avg_loss = 0.0f;
        hit = 0;
        shuffle_indices(order_indices, 60000);

        //learning section
        for (int loop = 0; loop < 60000; loop++){
            //if (!(loop%1000) && debug) {printf("%d datas have been processed.\n", loop);}

            //offset data
            fseek(learning_data_images, 16 + 784 * order_indices[loop], SEEK_SET);
            fseek(learning_data_labels, 8 + order_indices[loop], SEEK_SET);

            //inputting data
            for (int i = 0; i < n_of_input_layer; i++){
                input_layer[i] = (float)(fgetc(learning_data_images))/256;
            }

            //inputting label
            answer = fgetc(learning_data_labels);

            start = clock();
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

            end = clock();
            if (neck_check) {printf("Forward: %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);}

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

            start = clock();
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

            end = clock();
            if (neck_check) {printf("Backward: %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);}

            for (int i = 0; i < n_of_first_hidden_layer; i++){
                grad_to_b1t[i] += (float)grad_to_b1[i]/epoch;
            }
            for (int i = 0; i < n_of_second_hidden_layer; i++){
                grad_to_b2t[i] += (float)grad_to_b2[i]/epoch;
            }
            for (int i = 0; i < n_of_third_hidden_layer; i++){
                grad_to_b3t[i] += (float)grad_to_b3[i]/epoch;
            }
            for (int i = 0; i < n_of_output_layer; i++){
                grad_to_b4t[i] += (float)grad_to_b4[i]/epoch;
            }
            for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++){
                grad_to_w1t[i] += (float)grad_to_w1[i]/epoch;
            }
            for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++){
                grad_to_w2t[i] += (float)grad_to_w2[i]/epoch;
            }
            for (int i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++){
                grad_to_w3t[i] += (float)grad_to_w3[i]/epoch;
            }
            for (int i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++){
                grad_to_w4t[i] += (float)grad_to_w4[i]/epoch;
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
        printf("at epoch%d, learning has finished. average loss:%f\n", epoch_loop+1, avg_loss / 60000);
        fseek(learning_data_images, -784 * 60000, SEEK_CUR);
        fseek(learning_data_labels, -60000, SEEK_CUR);
        avg_loss = 0.0f;

        //testify section
        for (int loop = 0; loop < 10000; loop++){
            //if (!(loop%1000) && debug) {printf("%d datas have beed precessed.\n", loop);}
            
            //inputting data
            for (int i = 0; i < n_of_input_layer; i++)
            {
                input_layer[i] = (float)(fgetc(test_data_images))/256;
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
    fclose(learning_data_images);
    fclose(learning_data_labels);
    fclose(test_data_images);
    fclose(test_data_labels);
    return 0;
}