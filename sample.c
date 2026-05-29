#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "nnlib.h"

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

int CIFAR_10_loader (char *filename, float *image_buffer, uint8_t *answer_buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return 1;
    fseek(fp, 0, SEEK_SET);
    for (size_t i = 0; i < 10000; i++)
    {
        answer_buffer[i] = fgetc(fp);
        for (size_t j = 0; j < 32 * 32 * 3; j++)
        {
            image_buffer[i * 32*32*3 + j] = fgetc(fp)/255.0f;
        }
        
    }
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
    neural_network_t *nn = alloc_neural_network();
    add_conv_layer(nn, 32, 32, 3, 3, 3, 32, 1, 0);
    add_activation_layer(nn, LAYER_RELU);
    add_conv_layer(nn, 30, 30, 32, 3, 3, 32, 1, 0);
    add_activation_layer(nn, LAYER_RELU);
    add_pool_layer(nn, 28, 28, 32, 2, 2);
    add_conv_layer(nn, 14, 14, 32, 3, 3, 32, 1, 0);
    add_activation_layer(nn, LAYER_RELU);
    add_conv_layer(nn, 12, 12, 32, 3, 3, 32, 1, 0);
    add_activation_layer(nn, LAYER_RELU);
    add_pool_layer(nn, 10, 10, 32, 2, 2);
    add_fc_layer(nn, 5*5*32, 128);
    add_activation_layer(nn, LAYER_RELU);
    add_fc_layer(nn, 128, 10);
    add_activation_layer(nn, LAYER_SOFTMAX);
    
    parameter_initialize(nn);

    float *input_buffer = calloc(32 * 32 * 3 * 10000, sizeof(float));
    uint8_t *answer_label_buffer = calloc(10000, sizeof(uint8_t));
    printf("train data loaded\n");

    float *input_one_image = calloc(32 * 32 * 3, sizeof(float));
    float answer_one_label[10];

    float output[10];

    int hit = 0;
    int t = 0;

    printf("training start\n");
    for (int epoch = 0; epoch < 5; epoch++){
        for (int x = 0; x < 5; x++){
            if (x==0)
            {
                CIFAR_10_loader("data_batch_1.bin", input_buffer, answer_label_buffer);
            }
            else if (x == 1)
            {
                CIFAR_10_loader("data_batch_2.bin", input_buffer, answer_label_buffer);
            }
            else if (x == 2)
            {
                CIFAR_10_loader("data_batch_3.bin", input_buffer, answer_label_buffer);
            }
            else if (x == 3)
            {
                CIFAR_10_loader("data_batch_4.bin", input_buffer, answer_label_buffer);
            }
            else if (x == 4)
            {
                CIFAR_10_loader("data_batch_5.bin", input_buffer, answer_label_buffer);
            }
            for (size_t i = 0; i < 10000; i++)
            {
                memcpy(input_one_image, &input_buffer[32*32*3 * i], 32*32*3 * sizeof(float));
                for (size_t j = 0; j < 10; j++)
                {
                    answer_one_label[j] = 0.0 + (answer_label_buffer[i] == j);
                }

                forward_pass(nn, input_one_image);
                backward_pass(nn, input_one_image, answer_one_label);
                if ((i%50) == 49)
                {
                    t++;
                    update_param_adam(nn, learning_rate, regularization_rate, 0.9f, 0.999f, 1e-7, t, 50);
                }
            }
            printf("1data_batch finished\n");
        }
        printf("1epoch finished\n");
    }
    printf("training finished\n");

    CIFAR_10_loader("test_batch.bin", input_buffer, answer_label_buffer);

    for (size_t i = 0; i < 10000; i++)
    {
        memcpy(input_one_image, &input_buffer[32*32*3 * i], 32*32*3 * sizeof(float));
        for (size_t j = 0; j < 10; j++)
        {
            answer_one_label[j] = 0.0 + (answer_label_buffer[i] == j);
        }
        forward_pass(nn, input_one_image);
        if (answer_label_buffer[i] == find_max_index(nn->layers[nn->n_layers - 1].output, 10))
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