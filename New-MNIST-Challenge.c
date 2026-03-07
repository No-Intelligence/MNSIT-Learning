#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define num_input_units 10
#define number_of_layers_setting 3
#define loss_function_setting cross_ententropy
#define learning_rate_setting 0.001
#define batch_size_setting 32
#define epoch_setting 10
#define regularization_rate_setting 0.0001
#define dropout_rate_setting 0.5
#define layer_construction_setting {256, 128, 10}
#define activation_function_setting {relu, relu, softmax}

typedef enum {
    relu,
    softmax
} activation_func_t;

typedef enum {
    cross_ententropy
} loss_func_t;
typedef struct {
    //layer info
    activation_func_t activation_function;  //活性化関数の種類
    float *weight;                          //重み
    float *bias;                            //バイアス

    //buffer
    float *activation;      //ニューロンのアクティベーション
    float *pre_activation;  //ニューロンの重み付き和
    float *delta;           //勾配計算時のバッファ
    float *grad_w;          //重みの勾配のバッファ
    float *grad_b;          //バイアスの勾配のバッファ
}layer_t;

typedef struct {
    int num_layers;             //層の数を中間層、出力層の合計で入力
    loss_func_t loss_function;  //損失関数を指定
    float learning_rate;        //学習率
    int batch_size;             //バッチサイズ
    int epoch;                  //エポック数
    float regularization_rate;  //正規化レート
    float dropout_rate;         //ドロップアウト率
    layer_t *layers;            //層の配列へのポインタ
}network_t;

network_t* create_network(int num_layers, loss_func_t loss_func, float learning_rate, int batch_size, int epoch, float regularization_rate, float dropout_rate, int *layer_sizes, int num_first_layer, activation_func_t *activation_function) {
    network_t *temporary_internal_net = malloc(sizeof(network_t));
    temporary_internal_net->num_layers = num_layers;
    temporary_internal_net->loss_function = loss_func;
    temporary_internal_net->learning_rate = learning_rate;
    temporary_internal_net->batch_size = batch_size;
    temporary_internal_net->epoch = epoch;
    temporary_internal_net->regularization_rate = regularization_rate;
    temporary_internal_net->dropout_rate = dropout_rate;
    temporary_internal_net->layers = malloc(num_layers * sizeof(layer_t));
    for (size_t i = 0; i < num_layers; i++) {
        temporary_internal_net->layers[i].activation_function = activation_function[i];
        if (i == 0) temporary_internal_net->layers[i].weight = (float*)malloc(num_first_layer * layer_sizes[i] * sizeof(float));
        else temporary_internal_net->layers[i].weight = (float*)malloc(layer_sizes[i - 1] * layer_sizes[i] * sizeof(float));
        temporary_internal_net->layers[i].bias = (float*)malloc(layer_sizes[i] * sizeof(float));
        temporary_internal_net->layers[i].activation = (float*)malloc(layer_sizes[i] * sizeof(float));
        temporary_internal_net->layers[i].pre_activation = (float*)malloc(layer_sizes[i] * sizeof(float));
        temporary_internal_net->layers[i].delta = (float*)malloc(layer_sizes[i] * sizeof(float));
        if (i == 0) temporary_internal_net->layers[i].grad_w = (float*)malloc(num_first_layer * layer_sizes[i] * sizeof(float));
        else temporary_internal_net->layers[i].grad_w = (float*)malloc(layer_sizes[i - 1] * layer_sizes[i] * sizeof(float));
        temporary_internal_net->layers[i].grad_b = (float*)malloc(layer_sizes[i] * sizeof(float));
    }
    return temporary_internal_net;
}

void free_network (network_t *temporary_internal_net, int num_layers) {
    for (size_t i = 0; i < num_layers; i++)
    {
        free(temporary_internal_net->layers[i].weight);
        free(temporary_internal_net->layers[i].bias);
        free(temporary_internal_net->layers[i].activation);
        free(temporary_internal_net->layers[i].pre_activation);
        free(temporary_internal_net->layers[i].delta);
        free(temporary_internal_net->layers[i].grad_w);
        free(temporary_internal_net->layers[i].grad_b);
    }
    free(temporary_internal_net->layers);
    free(temporary_internal_net);
}

int load_MNIST_format_image (char filename, int num, float *buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return 1;
    fseek(fp, 16, SEEK_SET);
    for (size_t i = 0; i < 784 * num; i++)
    {
        buffer[i] = fgetc(fp)/255.0f;
    }
    fclose(fp);
    return 0;
}

int load_MNIST_format_label (char filename, int num, uint8_t *buffer) {
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL) return 1;
    fseek(fp, 8, SEEK_SET);
    fread(buffer, sizeof(uint8_t), num, fp);
    fclose(fp);
    return 0;
}

void weight_initialization (network_t *temporary_internal_net) {

}

int main(int argc, char const *argv[])
{
    int num_layers[number_of_layers_setting] = layer_construction_setting;
    activation_func_t activation_func[number_of_layers_setting] = activation_function_setting;
    network_t *neural_network = create_network(number_of_layers_setting, loss_function_setting, learning_rate_setting, batch_size_setting, epoch_setting, regularization_rate_setting, dropout_rate_setting, num_layers, num_input_units, activation_func);

    //file load
    FILE *train_image, *train_label, *test_image, *test_label;
    train_image = fopen("train-images.idx3-ubyte", "rb");
    train_label = fopen("train-labels.idx1-ubyte", "rb");
    test_image = fopen("t10k-images.idx3-ubyte", "rb");
    test_label = fopen("t10k-labels.idx1-ubyte", "rb");
    if (train_image == NULL) {
        printf("train image error\n");
        return 1;
    }
    if (train_label == NULL) {
        printf("train label error\n");
        return 2;
    }
    if (test_image == NULL) {
        printf("test image error\n");
        return 3;
    }
    if (test_label == NULL) {
        printf("test label error\n");
        return 4;
    }
    float *train_image_buffer = (float*)malloc(60000 * 784 * sizeof(float));
    uint8_t *train_label_buffer = (uint8_t*)malloc(60000 * sizeof(uint8_t));
    float *test_image_buffer = (float*)malloc(60000 * 784 * sizeof(float));
    uint8_t *test_label_buffer = (uint8_t*)malloc(60000 * sizeof(uint8_t));
    fseek(train_image, 16, SEEK_SET);
    fseek(train_label, 8, SEEK_SET);
    fseek(test_image, 16, SEEK_SET);
    fseek(test_label, 8, SEEK_SET);
    for (size_t i = 0; i < 60000 * 784; i++)
    {
        train_image_buffer[i] = fgetc(train_image)/255.0f;
    }
    for (size_t i = 0; i < 60000; i++)
    {
        train_label_buffer[i] = fgetc(train_label);
    }
    for (size_t i = 0; i < 10000 * 784; i++)
    {
        test_image_buffer[i] = fgetc(test_image)/255.0f;
    }
    for (size_t i = 0; i < 10000; i++)
    {
        test_label_buffer[i] = fgetc(test_label);
    }
    
    
    
    


    free_network(neural_network, number_of_layers_setting);
    free(train_image_buffer);
    free(train_label_buffer);
    free(test_image_buffer);
    free(test_label_buffer);
    return 0;
}
