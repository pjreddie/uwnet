#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols == b.cols);

    matrix y = copy_matrix(xw);
    int i,j;
    for(i = 0; i < xw.rows; ++i){
        for(j = 0; j < xw.cols; ++j){
            y.data[i*y.cols + j] += b.data[j];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_bias(matrix dy)
{
    matrix db = make_matrix(1, dy.cols);
    int i, j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = xw+b
matrix forward_connected_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    // TODO: 3.1 - run the network forward
    matrix y = make_matrix(x.rows, l.w.cols); // Going to want to change this!


    return y;
}

// Run a connected layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_connected_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    // TODO: 3.2
    // Calculate the gradient dL/db for the bias terms using backward_bias
    // add this into any stored gradient info already in l.db

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw

    // Calculate dL/dx and return it
    matrix dx = copy_matrix(x); // Change this


    return dx;
}

// Update weights and biases of connected layer
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_connected_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 3.3
    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little

    // Do the same for biases as well but no need to use weight decay on biases
}

layer make_connected_layer(int inputs, int outputs)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

