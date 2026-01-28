#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mllib.h"
#include "bmp_to_mnist.h"

#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_output_layer 10

int main(int argc, char const *argv[])
{
    if (argc > 2)
    {
        printf("too many images\n");
        return 4;
    }
    
    int answer = 0, hit = 0;
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
    int temp[n_of_input_layer];
    FILE *param, *test_data_images, *test_data_labels;

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
    
    param = fopen("params.bin", "r");
    if (param == NULL)
    {
        printf("err1");
        return 1;
    }
    
    test_data_images = fopen("t10k-images.idx3-ubyte", "rb");
    if (test_data_images == NULL)
    {
        printf("err2");
        return 2;
    }
    
    test_data_labels = fopen("t10k-labels.idx1-ubyte", "rb");
    if (test_data_labels == NULL)
    {
        printf("err3");
        return 3;
    }
    

    for (int i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
    {
        fscanf(param, "%f\n", &weight_to_first_hidden_layer[i]);
    }
    for (int i = 0; i < n_of_first_hidden_layer; i++)
    {
        fscanf(param, "%f\n", &bias_of_first_hidden_layer[i]);
    }
    for (int i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++)
    {
        fscanf(param, "%f\n", &weight_to_second_hidden_layer[i]);
    }
    for (int i = 0; i < n_of_second_hidden_layer; i++)
    {
        fscanf(param, "%f\n", &bias_of_second_hidden_layer[i]);
    }
    for (int i = 0; i < n_of_second_hidden_layer * n_of_output_layer; i++)
    {
        fscanf(param, "%f\n", &weight_to_output_layer[i]);
    }
    for (int i = 0; i < n_of_output_layer; i++)
    {
        fscanf(param, "%f\n", &bias_of_output_layer[i]);
    }
    
    //offset data
    fseek(test_data_images, 16, 0);
    fseek(test_data_labels, 8, 0);

    //testify section
    /*for (int loop = 0; loop < 10000; loop++){
    if (!(loop%100))
    {
        printf("%d datas have beed precessed.\n", loop);
    }
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

    mmul(zout, second_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_second_hidden_layer);
    add_bias(zout, bias_of_output_layer, n_of_output_layer);
    softmax(zout, output_layer, n_of_output_layer);

    if (find_max_index(output_layer, n_of_output_layer) == answer)
    {
        ++hit;
    }
    }
    printf("hit :%d, Hit rate is %f%%\n\n", hit, (float)hit/100);*/

    uint8_t data[MNIST_ARRAY_SIZE];
    bmp_to_mnist_array("unique", data);
    for (int i = 0; i < n_of_input_layer; i++)
    {
        input_layer[i] = (float)data[i]/255;
    }
    

    mmul(z1, input_layer, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
    add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
    relu(z1, first_hidden_layer, n_of_first_hidden_layer);

    mmul(z2, first_hidden_layer, weight_to_second_hidden_layer, n_of_second_hidden_layer, n_of_first_hidden_layer);
    add_bias(z2, bias_of_second_hidden_layer, n_of_second_hidden_layer);
    relu(z2, second_hidden_layer, n_of_second_hidden_layer);

    mmul(zout, second_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_second_hidden_layer);
    add_bias(zout, bias_of_output_layer, n_of_output_layer);
    softmax(zout, output_layer, n_of_output_layer);

    printf("The number is %d.\n", find_max_index(output_layer, n_of_output_layer));

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
    fclose(param);
    fclose(test_data_images);
    fclose(test_data_labels);
    return 0;
}
