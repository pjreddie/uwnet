#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "uwnet.h"
#include "matrix.h"

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float accuracy_net(net m, data d)
{
    matrix p = forward_net(m, d.x);
    int i;
    int correct = 0;
    for (i = 0; i < d.y.rows; ++i) {
        if (max_index(d.y.data + i*d.y.cols, d.y.cols) == max_index(p.data + i*p.cols, p.cols)) ++correct;
    }
    free_matrix(p);
    return (float)correct / d.y.rows;
}

float cross_entropy_loss(matrix x, matrix y)
{
    assert(x.rows == y.rows);
    assert(x.cols == y.cols);
    int i;
    float sum = 0;
    for(i = 0; i < y.cols*y.rows; ++i){
        sum += -y.data[i]*log(x.data[i]);
    }
    return sum/y.rows;
}

matrix cross_entropy_derivative(matrix x, matrix y)
{
    assert(x.rows == y.rows);
    assert(x.cols == y.cols);
    matrix d = make_matrix(x.rows, x.cols);
    int i;
    for(i = 0; i < y.cols*y.rows; ++i){
        d.data[i] = x.data[i] - y.data[i];
    }
    return d;
}

void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay)
{
    srand(0);
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix yhat = forward_net(m, b.x);
        float err = cross_entropy_loss(yhat, b.y);
        matrix dy = cross_entropy_derivative(yhat, b.y);
        fprintf(stderr, "%06d: Loss: %f\n", e, err);
        backward_net(m, dy);
        update_net(m, rate/batch, momentum, decay);
        free_data(b);
        free_matrix(yhat);
        free_matrix(dy);
    }
}
