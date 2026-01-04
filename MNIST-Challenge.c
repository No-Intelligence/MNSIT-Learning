#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265358979
#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_output_layer 10

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

float gelu (float in){
    float out = 0.0;
    out = 0.5 * in * (1 + tanh(sqrt(2/PI) * (in + 0.044715 * pow(in, 3))));
    return out;
}

float softmax (float *in, float *out, int n){
    float max = 0.0, sum = 0.0;
    max = extract_max(in, n);
    for (int i = 0; i < n; i++)
    {
        out[i] = exp(in[i] - max);
        sum += out[i];
    }
    for (int j = 0; j < n; j++)
    {
        out[j] = out[j] / sum;
    }
    
    return 0;
    
}

int main (void){
    //define variables
    int answer = 0;

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

    //error
    if (!input_layer || !first_hidden_layer || !second_hidden_layer || !output_layer || !weight_to_first_hidden_layer || !bias_of_first_hidden_layer || !weight_to_second_hidden_layer || !bias_of_second_hidden_layer || !weight_to_output_layer || !bias_of_output_layer)
    {
        printf("Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    //file
    FILE *learning_data_images, *learning_data_labels;

    //weight initialize
    srand(9415);
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
    fclose(learning_data_images);
    fclose(learning_data_labels);
    return 0;
}