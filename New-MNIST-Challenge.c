#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

typedef struct {
    //layer info
    int input_units;        //入力ユニット数
    int output_units;       //出力ユニット数
    int activation_func;    //活性化関数の種類
    float *weight;          //重み
    float *bias;            //バイアス

    //buffer
    float *input;
    float *output;
    float *delta;
    float *grad_w;
    float *grad_b;
}layer_t;

typedef struct {
    int num_layers;         //層の数を入力層、中間層、出力層の合計で入力
    int loss_function;  //損失関数を指定(0で交差エントロピー)将来的に拡張
    layer_t *layers;    //層の配列へのポインタ
}network_t;

network_t* create_network(int num_layers, int loss_func, int *layer_sizes) {
    network_t *net = malloc(sizeof(network_t));
    net->num_layers = num_layers;
    net->loss_function = loss_func;
    net->layers = malloc(num_layers * sizeof(layer_t));

    for (size_t i = 0; i < num_layers; i++) {
    }
    
}

int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
