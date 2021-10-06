#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"

void try_hw0()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.n = 4;
    n.layers = calloc(n.n, sizeof(layer));
    n.layers[0] = make_connected_layer(784, 32);
    n.layers[1] = make_activation_layer(RELU);
    n.layers[2] = make_connected_layer(32, 10);
    n.layers[3] = make_activation_layer(SOFTMAX);

    int batch = 128;
    int iters = 1500;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
    free_data(train);
    free_data(test);
    free_net(n);
}

void try_hw1()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.n = 8;
    n.layers = calloc(n.n, sizeof(layer));
    n.layers[0] = make_convolutional_layer(28, 28, 1, 8, 3, 1);
    n.layers[1] = make_activation_layer(RELU);
    n.layers[2] = make_maxpool_layer(28, 28, 8, 3, 2);
    n.layers[3] = make_convolutional_layer(14, 14, 8, 16, 3, 1);
    n.layers[4] = make_activation_layer(RELU);
    n.layers[5] = make_maxpool_layer(14, 14, 16, 3, 2);
    n.layers[6] = make_connected_layer(784, 10);
    n.layers[7] = make_activation_layer(SOFTMAX);

    int batch = 128;
    int iters = 1500;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
    free_data(train);
    free_data(test);
    free_net(n);
}

int main(int argc, char **argv)
{
    if(argc < 2){
        printf("usage: %s [test | tryhw0 | tryhw1]\n", argv[0]);  
    } else if (0 == strcmp(argv[1], "tryhw0")){
        try_hw0();
    } else if (0 == strcmp(argv[1], "tryhw1")){
        try_hw1();
    } else if (0 == strcmp(argv[1], "test")){
        run_tests();
    }
    return 0;
}
