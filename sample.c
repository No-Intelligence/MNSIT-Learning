#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "fcnnlib.h"

int n_layers = 8;
int layer_size[] = {784, 512, 256, 128, 64, 32, 16, 10};
activation_t activations[] = {ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_SOFTMAX};
float learning_rate = 0.001f;
float regularization_rate = 0.0005f;

int load_MNIST_format_image (char *filename, int load_num, float *buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return 1;
    fseek(fp, 16, SEEK_SET);
    for (size_t i = 0; i < 784 * load_num; i++)
    {
        buffer[i] = fgetc(fp)/255.0f;
    }
    fclose(fp);
    return 0;
}

int load_MNIST_format_label (char *filename, int load_num, uint8_t *buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return 1;
    fseek(fp, 8, SEEK_SET);
    fread(buffer, sizeof(uint8_t), load_num, fp);
    fclose(fp);
    return 0;
}

int find_max_index (float *array, int length) {
    int max_index = 0;
    float f = array[0];
    for (size_t i = 0; i < length; i++)
    {
        if (f <= array[i])
        {
            f = array[i];
            max_index = i;
        }
    }
    return max_index;
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    neural_network_t *nn = alloc_neural_network(n_layers, layer_size, activations);
    parameter_initialize(nn);

    float *input_buffer = calloc(60000 * 784, sizeof(float));
    uint8_t *answer_label_buffer = calloc(60000, sizeof(uint8_t));
    load_MNIST_format_image("train-images-fashion.idx3-ubyte", 60000, input_buffer);
    load_MNIST_format_label("train-labels-fashion.idx1-ubyte", 60000, answer_label_buffer);

    float *input_one_image = calloc(784, sizeof(float));
    float answer_one_label[10];

    float output[10];

    int hit = 0;

    for (size_t i = 0; i < 60000; i++)
    {
        memcpy(input_one_image, &input_buffer[784 * i], 784 * sizeof(float));
        for (size_t j = 0; j < 10; j++)
        {
            answer_one_label[j] = 0.0 + (answer_label_buffer[i] == j);
        }

        forward_pass(nn, input_one_image, output);
        backward_pass(nn, answer_one_label);
        //update_param(nn, learning_rate, regularization_rate);
        update_param_adam(nn, learning_rate, regularization_rate, 0.9f, 0.999f, 1e-7, i);

        if (i % 3000 == 0) {
            printf("%.1f%%\n", (i / 60000.0f) * 100.0f);
        }
    }

    load_MNIST_format_image("t10k-images-fashion.idx3-ubyte", 10000, input_buffer);
    load_MNIST_format_label("t10k-labels-fashion.idx1-ubyte", 10000, answer_label_buffer);


    for (size_t i = 0; i < 10000; i++)
    {
        memcpy(input_one_image, &input_buffer[784 * i], 784 * sizeof(float));
        for (size_t j = 0; j < 10; j++)
        {
            answer_one_label[j] = 0.0 + (answer_label_buffer[i] == j);
        }
        forward_pass(nn, input_one_image, output);
        if (answer_label_buffer[i] == find_max_index(output, 10))
        {
            hit++;
        }
        
    }
    printf("%f%%\n", ((float)hit / 10000.0f) * 100.0f);

    free(input_buffer);
    free(answer_label_buffer);
    free(input_one_image);
    free_neural_network(nn);
    return 0;
}