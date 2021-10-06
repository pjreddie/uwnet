#include <stdlib.h>
#include <stdio.h>
#include "uwnet.h"

matrix forward_net(net m, matrix input)
{
    int i;
    matrix x = copy_matrix(input);
    for (i = 0; i < m.n; ++i) {
        layer l = m.layers[i];
        matrix y = l.forward(l, x);

        free_matrix(x);
        x = y;
    }
    return x;
}

void backward_net(net m, matrix d)
{
    matrix dy = copy_matrix(d);
    int i;
    for (i = m.n-1; i >= 0; --i) {
        layer l = m.layers[i];
        matrix dx = l.backward(l, dy);

        free_matrix(dy);
        dy = dx;
    }
    free_matrix(dy);
}

void update_net(net m, float rate, float momentum, float decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        l.update(l, rate, momentum, decay);
    }
}

void free_layer(layer l)
{
    free_matrix(l.w);
    free_matrix(l.dw);
    free_matrix(l.b);
    free_matrix(l.db);
    if(l.x){
        free_matrix(*l.x);
        free(l.x);
    }
}

void free_net(net n)
{
    int i;
    for(i = 0; i < n.n; ++i){
        free_layer(n.layers[i]);
    }
    free(n.layers);
}

void file_error(char *filename)
{
    fprintf(stderr, "Couldn't open file %s\n", filename);
    exit(-1);
}

void save_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) write_matrix(l.b, fp);
        if(l.w.data) write_matrix(l.w, fp);
    }
    fclose(fp);
}

void load_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) read_matrix(l.b, fp);
        if(l.w.data) read_matrix(l.w, fp);
    }
    fclose(fp);
}
