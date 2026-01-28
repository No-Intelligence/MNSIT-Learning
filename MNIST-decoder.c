#include <stdio.h>
#include "mllib.h"

#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_output_layer 10

int main(int argc, char const *argv[])
{
    float *input_layer, *first_hidden_layer, *second_hidden_layer, *output_layer, *z1, *z2,*zout, *weight_to_first_hidden_layer, *weight_to_second_hidden_layer, *weight_to_output_layer, *bias_of_first_hidden_layer, *bias_of_second_hidden_layer, *bias_of_output_layer;
    
    FILE *param, *test_data_images, *test_data_labels;
    param = fopen("params.bin", "r");
    test_data_images = fopen("t10k-images.idx3-ubyte", "rb");
    test_data_labels = fopen("t10k-labels.idx1-ubyte", "rb");







    //offset data
    fseek(test_data_images, 16, 0);
    fseek(test_data_labels, 8, 0);

    //testify section
    for (int loop = 0; loop < 10000; loop++){
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
    printf("hit :%d, Hit rate is %f%%\n", hit, (float)hit/100);
    return 0;
}
