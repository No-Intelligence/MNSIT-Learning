#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979
#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_output_layer 10

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
        output_arr[i] = max(input_arr[i], 0.0f);
    }
    
}

void gelu (float *input_arr, float *output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = 0.5 * input_arr[i] * (1 + tanh(sqrt(2/PI) * (input_arr[i] + 0.044715 * pow(input_arr[i], 3))));
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

void compute_output_delta (float *delta3, float *output_layer, float *answer_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++) {
        delta3[i] = output_layer[i] - answer_arr[i];
    }
}

void weight_grad (float *z, float *previous_activation, float *output_arr, int n_of_output, int n_of_input){
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < count; j++)
        {
            output_arr[i + j];
        }
        
    }
    
}

int main (void){
    //define variables
    int answer = 0;
    float answer_arr[n_of_output_layer] = {0.0};
    float loss = 0.0f;

    //define pointer
    float *input_layer;
    float *first_hidden_layer;
    float *second_hidden_layer;
    float *output_layer;
    float *weight_to_first_hidden_layer;
    float *bias_of_first_hidden_layer;
    float *weight_to_second_hidden_layer;
    float *bias_of_second_hidden_layer;
    float *weight_to_output_layer;
    float *bias_of_output_layer;
    float *z1;
    float *z2;
    float *zout;

    //allocetion params
    input_layer = (float*)malloc(n_of_input_layer * sizeof(float));
    first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    weight_to_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * n_of_input_layer * sizeof(float));
    bias_of_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    weight_to_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * n_of_first_hidden_layer * sizeof(float));
    bias_of_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    weight_to_output_layer = (float*)malloc(n_of_output_layer * n_of_second_hidden_layer * sizeof(float));
    bias_of_output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    z1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    z2 = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    zout = (float*)malloc(n_of_output_layer * sizeof(float));

    //error
    if (!input_layer || !first_hidden_layer || !second_hidden_layer || !output_layer || !weight_to_first_hidden_layer || !bias_of_first_hidden_layer || !weight_to_second_hidden_layer || !bias_of_second_hidden_layer || !weight_to_output_layer || !bias_of_output_layer)
    {
        printf("Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    //file
    FILE *learning_data_images, *learning_data_labels;

    //weight initialize
    srand(time(NULL));
    for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
    {
        weight_to_first_hidden_layer[i] = rand() / (RAND_MAX+1.0);
    }
    for (int i = 0; i < n_of_first_hidden_layer; i++)
    {
        bias_of_first_hidden_layer[i] = 0;
    }
    for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++)
    {
        weight_to_second_hidden_layer[i] = rand() / (RAND_MAX+1.0);
    }
    for (int i = 0; i < n_of_second_hidden_layer; i++)
    {
        bias_of_second_hidden_layer[i] = 0;
    }
    for (int i = 0; i < n_of_second_hidden_layer * n_of_output_layer; i++)
    {
        weight_to_output_layer[i] = rand() / (RAND_MAX+1.0);
    }
    for (int i = 0; i < n_of_output_layer; i++)
    {
        bias_of_output_layer[i] = 0;
    }


    //loading datas
    learning_data_images = fopen("train-images.idx3-ubyte", "rb");
    if (learning_data_images == NULL)
    {
        printf("images err\n");
        return 1;
    }
    printf("images have loaded successfully.\n");
    learning_data_labels = fopen("train-labels.idx1-ubyte", "rb");
    if (learning_data_labels == NULL)
    {
        printf("labels err\n");
        return 2;
    }
    printf("labels have loaded successfully\n");

    //offset data
    fseek(learning_data_images, 16, 0);
    fseek(learning_data_labels, 8, 0);

    //inputting data
    for (int i = 0; i < n_of_input_layer; i++)
    {
        input_layer[i] = (float)(fgetc(learning_data_images))/256;
    }

    //inputting label
    answer = fgetc(learning_data_labels);

    //forward pass
    mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
    add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
    relu(z1, first_hidden_layer, n_of_first_hidden_layer);

    mmul(z2, first_hidden_layer, weight_to_first_hidden_layer, n_of_second_hidden_layer, n_of_first_hidden_layer);
    add_bias(z2, bias_of_second_hidden_layer, n_of_second_hidden_layer);
    relu(z2, second_hidden_layer, n_of_second_hidden_layer);

    mmul(zout, second_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_second_hidden_layer);
    add_bias(zout, bias_of_output_layer, n_of_output_layer);
    softmax(zout, output_layer, n_of_output_layer);

    //loss function (cross entropy)
    for (int i = 0; i < n_of_output_layer; i++)
    {
        answer_arr[i] = 0.0;
    }
    answer_arr[answer] = 1.0;
    for (int i = 0; i < n_of_output_layer; i++)
    {
        loss += answer_arr[i] * logf(output_layer[i] + 1e-8f);
    }
    loss = -loss;
    printf("%f", loss);



    //end
    free(input_layer);
    free(first_hidden_layer);
    free(second_hidden_layer);
    free(output_layer);
    free(weight_to_first_hidden_layer);
    free(bias_of_first_hidden_layer);
    free(weight_to_second_hidden_layer);
    free(bias_of_second_hidden_layer);
    free(weight_to_output_layer);
    free(bias_of_output_layer);
    free(z1);
    free(z2);
    free(zout);
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
    fclose(learning_data_images);
    fclose(learning_data_labels);
    return 0;
}