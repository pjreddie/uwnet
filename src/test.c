#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "uwnet.h"
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"

// Forward declare for tests
matrix mean(matrix x, int groups);
matrix variance(matrix x, matrix m, int groups);
matrix normalize(matrix x, matrix m, matrix v, int groups);
matrix delta_mean(matrix d, matrix v);
matrix delta_variance(matrix d, matrix x, matrix m, matrix v);
matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix m, matrix v, matrix x);

int tests_total = 0;
int tests_fail = 0;


int within_eps(float a, float b){
    return a-EPS<b && b<a+EPS;
}

int same_matrix(matrix a, matrix b)
{
    int i;
    if(a.rows != b.rows || a.cols != b.cols) {
        printf ("first matrix: %dx%d, second matrix:%dx%d\n", a.rows, a.cols, b.rows, b.cols);
        return 0;
    }
    for(i = 0; i < a.rows*a.cols; ++i){
        if(!within_eps(a.data[i], b.data[i])) {
            printf("differs at %d, %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_copy_matrix()
{
    matrix a = random_matrix(32, 64, 10);
    matrix c = copy_matrix(a);
    TEST(same_matrix(a,c));
    free_matrix(a);
    free_matrix(c);
}

void test_transpose_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix at = load_matrix("data/test/at.matrix");
    matrix atest = transpose_matrix(a);
    matrix aorig = transpose_matrix(atest);
    TEST(same_matrix(at, atest) && same_matrix(a, aorig));
    free_matrix(a);
    free_matrix(at);
    free_matrix(atest);
    free_matrix(aorig);
}

void test_axpy_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix y1 = load_matrix("data/test/y1.matrix");
    axpy_matrix(2, a, y);
    TEST(same_matrix(y, y1));
    free_matrix(a);
    free_matrix(y);
    free_matrix(y1);
}

void test_matmul()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix c = load_matrix("data/test/c.matrix");
    matrix mul = matmul(a, b);
    TEST(same_matrix(c, mul));
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(mul);
}

void test_activation_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");

    layer log_layer = make_activation_layer(LOGISTIC);
    layer relu_layer = make_activation_layer(RELU);
    layer lrelu_layer = make_activation_layer(LRELU);
    layer soft_layer = make_activation_layer(SOFTMAX);

    matrix alog = log_layer.forward(log_layer, a);
    matrix arelu = relu_layer.forward(relu_layer, a);
    matrix alrelu = lrelu_layer.forward(lrelu_layer, a);
    matrix asoft = soft_layer.forward(soft_layer, a);

    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));

    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");

    matrix glog = log_layer.backward(log_layer, y);
    matrix grelu = relu_layer.backward(relu_layer, y);
    matrix glrelu = lrelu_layer.backward(lrelu_layer, y);
    matrix gsoft = soft_layer.backward(soft_layer, y);

    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));

    free_matrix(a);
    free_matrix(y);
    free_matrix(alog);
    free_matrix(arelu);
    free_matrix(alrelu);
    free_matrix(asoft);
    free_matrix(glog);
    free_matrix(grelu);
    free_matrix(glrelu);
    free_matrix(gsoft);
    free_matrix(truth_alog);
    free_matrix(truth_arelu);
    free_matrix(truth_alrelu);
    free_matrix(truth_asoft);
    free_matrix(truth_glog);
    free_matrix(truth_grelu);
    free_matrix(truth_glrelu);
    free_matrix(truth_gsoft);
    free_layer(log_layer);
    free_layer(relu_layer);
    free_layer(lrelu_layer);
    free_layer(soft_layer);
}

void test_connected_layer()
{
    matrix x = load_matrix("data/test/a.matrix");
    matrix w = load_matrix("data/test/b.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix db = load_matrix("data/test/db.matrix");
    matrix dy = load_matrix("data/test/dy.matrix");
    matrix truth_dx = load_matrix("data/test/truth_dx.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");
    matrix truth_db = load_matrix("data/test/truth_db.matrix");
    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_db = load_matrix("data/test/updated_db.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_b = load_matrix("data/test/updated_b.matrix");

    matrix b = load_matrix("data/test/bias.matrix");
    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_connected_layer(64, 16);
    free_matrix(l.w);
    free_matrix(l.b);
    free_matrix(l.dw);
    free_matrix(l.db);
    l.w = w;
    l.b = b;
    l.dw = dw;
    l.db = db;
    matrix out = l.forward(l, x);
    TEST(same_matrix(truth_out, out));

    matrix dx = l.backward(l, dy);
    TEST(same_matrix(truth_dx, dx));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_db, l.db));

    l.update(l, 1, .9, .5);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_db, l.db));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_b, l.b));

    free_matrix(x);
    free_matrix(dx);
    free_matrix(dy);
    free_matrix(out);
    free_matrix(truth_out);
    free_layer(l);
    free_matrix(truth_dx);
    free_matrix(truth_db);
    free_matrix(truth_dw);
    free_matrix(updated_db);
    free_matrix(updated_dw);
    free_matrix(updated_b);
    free_matrix(updated_w);
}

void test_im2col()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix truth_col = load_matrix("data/test/im2col.matrix");
    matrix col2 = im2col(im, 2, 2);
    matrix truth_col2 = load_matrix("data/test/im2col2.matrix");
    TEST(same_matrix(truth_col,   col));
    TEST(same_matrix(truth_col2,  col2));
    free_matrix(col);
    free_matrix(col2);
    free_matrix(truth_col);
    free_matrix(truth_col2);
    free_image(im);
}

void test_col2im()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix dcol = load_matrix("data/test/dcol.matrix");
    matrix dcol2 = load_matrix("data/test/dcol2.matrix");
    image col2im_res = col2im(im.w, im.h, im.c, dcol, 3, 2);
    image col2im_res2 = col2im(im.w, im.h, im.c, dcol2, 2, 2);

    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;

    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;

    matrix truth_col2mat = load_matrix("data/test/col2mat.matrix");
    matrix truth_col2mat2 = load_matrix("data/test/col2mat2.matrix");
    TEST(same_matrix(truth_col2mat, col2mat));
    TEST(same_matrix(truth_col2mat2, col2mat2));
    free_matrix(dcol);
    free_matrix(col2mat);
    free_matrix(truth_col2mat);
    free_matrix(dcol2);
    free_matrix(col2mat2);
    free_matrix(truth_col2mat2);
    free_image(im);
}


void test_maxpool_layer()
{
    image im = load_image("data/test/dog.jpg"); 

    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;

    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);

    matrix max_y = max_l.forward(max_l, im_mat);
    matrix max_y3 = max_l3.forward(max_l3, im_mat3);

    matrix truth_max_y = load_matrix("data/test/max_y.matrix");
    matrix truth_max_y3 = load_matrix("data/test/max_y3.matrix");

    TEST(same_matrix(truth_max_y, max_y));
    TEST(same_matrix(truth_max_y3, max_y3));

    matrix max_dy = load_matrix("data/test/max_dy.matrix");
    matrix max_dy3 = load_matrix("data/test/max_dy3.matrix");

    matrix max_dx = max_l.backward(max_l, max_dy);
    matrix max_dx3 = max_l3.backward(max_l3, max_dy3);

    matrix truth_max_dx = load_matrix("data/test/max_dx.matrix");
    matrix truth_max_dx3 = load_matrix("data/test/max_dx3.matrix");

    TEST(same_matrix(truth_max_dx, max_dx));
    TEST(same_matrix(truth_max_dx3, max_dx3));


    free_matrix(max_y);
    free_matrix(max_y3);
    free_matrix(truth_max_y);
    free_matrix(truth_max_y3);
    free_matrix(max_dx);
    free_matrix(max_dx3);
    free_matrix(max_dy);
    free_matrix(max_dy3);
    free_matrix(truth_max_dx);
    free_matrix(truth_max_dx3);
    free_image(im);
    free_layer(max_l);
    free_layer(max_l3);
}

void test_batchnorm_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");

    matrix mu_a   = mean(a, 64);
    matrix mu_a_s = mean(a, 8);
    matrix sig_a   =  variance(a, mu_a, 64);
    matrix sig_a_s =  variance(a, mu_a_s, 8);
    matrix norm_a = normalize(a, mu_a, sig_a, 64);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    matrix truth_mu_a = load_matrix("data/test/mu_a.matrix");
    matrix truth_mu_a_s = load_matrix("data/test/mu_a_s.matrix");
    matrix truth_sig_a = load_matrix("data/test/sig_a.matrix");
    matrix truth_sig_a_s = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_norm_a = load_matrix("data/test/norm_a.matrix");
    matrix truth_norm_a_s = load_matrix("data/test/norm_a_s.matrix");

    TEST(same_matrix(truth_mu_a,   mu_a));
    TEST(same_matrix(truth_mu_a_s, mu_a_s));
    TEST(same_matrix(truth_sig_a,   sig_a));
    TEST(same_matrix(truth_sig_a_s, sig_a_s));
    TEST(same_matrix(truth_norm_a,   norm_a));
    TEST(same_matrix(truth_norm_a_s, norm_a_s));

    

    matrix dm = delta_mean(y, sig_a);
    matrix dm_s = delta_mean(y, sig_a_s);

    matrix dv = delta_variance(y, a, mu_a, sig_a);
    matrix dv_s = delta_variance(y, a, mu_a_s, sig_a_s);

    matrix dbn = delta_batch_norm(y, dm, dv, mu_a, sig_a, a);
    matrix dbn_s = delta_batch_norm(y, dm_s, dv_s, mu_a_s, sig_a_s, a);

    matrix truth_dm   = load_matrix("data/test/dm.matrix");
    matrix truth_dm_s = load_matrix("data/test/dm_s.matrix");
    matrix truth_dv   = load_matrix("data/test/dv.matrix");
    matrix truth_dv_s = load_matrix("data/test/dv_s.matrix");
    matrix truth_dbn   = load_matrix("data/test/dbn.matrix");
    matrix truth_dbn_s = load_matrix("data/test/dbn_s.matrix");
    TEST(same_matrix(truth_dm,   dm));
    TEST(same_matrix(truth_dm_s, dm_s));
    TEST(same_matrix(truth_dv,   dv));
    TEST(same_matrix(truth_dv_s, dv_s));
    TEST(same_matrix(truth_dbn,   dbn));
    TEST(same_matrix(truth_dbn_s, dbn_s));

    free_matrix(truth_mu_a);
    free_matrix(truth_mu_a_s);
    free_matrix(truth_sig_a);
    free_matrix(truth_sig_a_s);
    free_matrix(truth_norm_a);
    free_matrix(truth_norm_a_s);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
    free_matrix(norm_a);
    free_matrix(norm_a_s);
    free_matrix(truth_dm);
    free_matrix(truth_dm_s);
    free_matrix(truth_dv);
    free_matrix(truth_dv_s);
    free_matrix(truth_dbn);
    free_matrix(truth_dbn_s);
    free_matrix(dm);
    free_matrix(dm_s);
    free_matrix(dv);
    free_matrix(dv_s);
    free_matrix(dbn);
    free_matrix(dbn_s);
    free_matrix(a);
    free_matrix(y);
}

void make_matrix_test()
{
    srand(1);
    matrix a = random_matrix(32, 64, 10);
    matrix b = random_matrix(64, 16, 10);
    matrix at = transpose_matrix(a);
    matrix c = matmul(a, b);
    matrix y = random_matrix(32, 64, 10);
    matrix bias = random_matrix(1, 16, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix db = random_matrix(1, 16, 10);
    matrix dy = random_matrix(32, 16, 10);
    matrix y1 = copy_matrix(y);
    axpy_matrix(2, a, y1);
    save_matrix(a, "data/test/a.matrix");
    save_matrix(b, "data/test/b.matrix");
    save_matrix(bias, "data/test/bias.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(db, "data/test/db.matrix");
    save_matrix(at, "data/test/at.matrix");
    save_matrix(dy, "data/test/dy.matrix");
    save_matrix(c, "data/test/c.matrix");
    save_matrix(y, "data/test/y.matrix");
    save_matrix(y1, "data/test/y1.matrix");

    layer log_layer = make_activation_layer(LOGISTIC);
    layer relu_layer = make_activation_layer(RELU);
    layer lrelu_layer = make_activation_layer(LRELU);
    layer soft_layer = make_activation_layer(SOFTMAX);

    matrix alog = log_layer.forward(log_layer, a);
    matrix arelu = relu_layer.forward(relu_layer, a);
    matrix alrelu = lrelu_layer.forward(lrelu_layer, a);
    matrix asoft = soft_layer.forward(soft_layer, a);

    matrix glog = log_layer.backward(log_layer, y);
    matrix grelu = relu_layer.backward(relu_layer, y);
    matrix glrelu = lrelu_layer.backward(lrelu_layer, y);
    matrix gsoft = soft_layer.backward(soft_layer, y);

    save_matrix(alog, "data/test/alog.matrix");
    save_matrix(arelu, "data/test/arelu.matrix");
    save_matrix(alrelu, "data/test/alrelu.matrix");
    save_matrix(asoft, "data/test/asoft.matrix");

    save_matrix(glog, "data/test/glog.matrix");
    save_matrix(grelu, "data/test/grelu.matrix");
    save_matrix(glrelu, "data/test/glrelu.matrix");
    save_matrix(gsoft, "data/test/gsoft.matrix");


    layer l = make_connected_layer(64, 16);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;

    matrix out = l.forward(l, a);
    save_matrix(out, "data/test/out.matrix");

    matrix dx = l.backward(l, dy);
    save_matrix(dx, "data/test/truth_dx.matrix");
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.db, "data/test/truth_db.matrix");

    l.update(l, 1, .9, .5);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.db, "data/test/updated_db.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.b, "data/test/updated_b.matrix");

    // Maxpool Layer Tests

    image im = load_image("data/test/dog.jpg"); 

    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;

    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);

    matrix max_y = max_l.forward(max_l, im_mat);
    matrix max_y3 = max_l3.forward(max_l3, im_mat3);

    save_matrix(max_y, "data/test/max_y.matrix");
    save_matrix(max_y3, "data/test/max_y3.matrix");

    matrix max_dy = random_matrix(max_y.rows, max_y.cols, 10);
    matrix max_dy3 = random_matrix(max_y3.rows, max_y3.cols, 10);

    save_matrix(max_dy, "data/test/max_dy.matrix");
    save_matrix(max_dy3, "data/test/max_dy3.matrix");

    matrix max_dx = max_l.backward(max_l, max_dy);
    matrix max_dx3 = max_l3.backward(max_l3, max_dy3);

    save_matrix(max_dx, "data/test/max_dx.matrix");
    save_matrix(max_dx3, "data/test/max_dx3.matrix");



    // im2col tests

    //image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix col2 = im2col(im, 2, 2);
    save_matrix(col, "data/test/im2col.matrix");
    save_matrix(col2, "data/test/im2col2.matrix");

    matrix dcol = random_matrix(col.rows, col.cols, 10);
    matrix dcol2 = random_matrix(col2.rows, col2.cols, 10);
    image col2im_res = col2im(im.w, im.h, im.c, dcol, 3, 2);
    image col2im_res2 = col2im(im.w, im.h, im.c, dcol2, 2, 2);
    save_matrix(dcol, "data/test/dcol.matrix");
    save_matrix(dcol2, "data/test/dcol2.matrix");
    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;
    save_matrix(col2mat, "data/test/col2mat.matrix");
    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;
    save_matrix(col2mat2, "data/test/col2mat2.matrix");



    // Batch norm test

    matrix mu_a   = mean(a, 64);
    matrix mu_a_s = mean(a, 8);

    matrix sig_a   =  variance(a, mu_a, 64);
    matrix sig_a_s =  variance(a, mu_a_s, 8);

    matrix norm_a = normalize(a, mu_a, sig_a, 64);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    save_matrix(mu_a, "data/test/mu_a.matrix");
    save_matrix(mu_a_s, "data/test/mu_a_s.matrix");
    save_matrix(sig_a, "data/test/sig_a.matrix");
    save_matrix(sig_a_s, "data/test/sig_a_s.matrix");
    save_matrix(norm_a, "data/test/norm_a.matrix");
    save_matrix(norm_a_s, "data/test/norm_a_s.matrix");

    matrix dm = delta_mean(y, sig_a);
    matrix dm_s = delta_mean(y, sig_a_s);

    save_matrix(dm, "data/test/dm.matrix");
    save_matrix(dm_s, "data/test/dm_s.matrix");

    matrix dv = delta_variance(y, a, mu_a, sig_a);
    matrix dv_s = delta_variance(y, a, mu_a_s, sig_a_s);

    save_matrix(dv, "data/test/dv.matrix");
    save_matrix(dv_s, "data/test/dv_s.matrix");

    matrix dbn = delta_batch_norm(y, dm, dv, mu_a, sig_a, a);
    matrix dbn_s = delta_batch_norm(y, dm_s, dv_s, mu_a_s, sig_a_s, a);
    save_matrix(dbn, "data/test/dbn.matrix");
    save_matrix(dbn_s, "data/test/dbn_s.matrix");
}


void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void run_hw0_tests()
{
    test_copy_matrix();
    test_axpy_matrix();
    test_transpose_matrix();
    test_matmul();
    test_activation_layer();
    test_connected_layer();
}

void run_hw1_tests()
{
    test_im2col();
    test_col2im();
    test_maxpool_layer();
}

void run_hw2_tests()
{
    test_batchnorm_layer();
}

void run_tests()
{
    run_hw0_tests();
    run_hw1_tests();
    run_hw2_tests();
}

void make_tests()
{
    // Don't run this function, it generates the ground truth matrices to test code against
    // make_matrix_test();
}

