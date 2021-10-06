// Include guards and C++ compatibility
#ifndef UWNET_H
#define UWNET_H
#include "image.h"
#include "matrix.h"
#ifdef __cplusplus
extern "C" {
#endif

// Layer and network definitions

// The kinds of activations our framework supports
typedef enum{LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX} ACTIVATION;

typedef struct layer {
    matrix *x;

    // Weights
    matrix w;
    matrix dw;

    // Biases
    matrix b;
    matrix db;

    // Image dimensions
    int width, height, channels;
    int size, stride, filters;
    ACTIVATION activation;

    // Batch norm matrices
    int batchnorm;
    matrix x_norm;
    matrix rolling_mean;
    matrix rolling_variance;

    matrix  (*forward)  (struct layer, struct matrix);
    matrix  (*backward) (struct layer, struct matrix);
    void   (*update)   (struct layer, float rate, float momentum, float decay);
} layer;

layer make_connected_layer(int inputs, int outputs);
layer make_activation_layer(ACTIVATION activation);
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride);
layer make_maxpool_layer(int w, int h, int c, int size, int stride);
layer make_batchnorm_layer(int groups);


typedef struct {
    layer *layers;
    int n;
} net;

matrix forward_net(net m, matrix x);
void backward_net(net m, matrix d);
void update_net(net m, float rate, float momentum, float decay);
void free_layer(layer l);
void free_net(net n);

typedef struct{
    matrix x;
    matrix y;
} data;
data random_batch(data d, int n);
data load_image_classification_data(char *images, char *label_file);
void free_data(data d);
void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay);
float accuracy_net(net m, data d);

char *fgetl(FILE *fp);

matrix im2col(image im, int size, int stride);
image col2im(int width, int height, int channels, matrix col, int size, int stride);

#ifdef __cplusplus
}
#endif
#endif
