---
layout: notes
title: Test Space
---

## Neural Network Example
{% include chart
chart='
graph LR
    x1(("\(x_1\)")) -->|"\(w_1\)"| y1(("\(y_1\)"))
    x2(("\(x_2\)")) -->|"\(w_2\)"| y1
    x3(("\(x_3\)")) -->|"\(w_3\)"| y1
    x4(("\(x_4\)")) -->|"\(w_4\)"| y1
    x1 --> |"\(w_5\)"| y2(("\(y_1\)"))
    x2 --> |"\(w_6\)"| y2
    x3 --> |"\(w_7\)"| y2
    x4 --> |"\(w_8\)"| y2
    y1 --> |"\(v_1\)"| z(("\(z\)"))
    y2 --> |"\(v_2\)"| z
'
caption="An example, 2 layer neural network. The first layer is computed as \(y = wx\). The second layer is computed as \(z = vy\)"
%}

## Loss Functions
{% include chart
chart='
graph LR
    a["model parameters <br /> (weights)"] --> model
    subgraph Loss Function
    model --> | predictions | error
    data --> | labels | error
    data --> | features | model
    end
    error --> loss
'
caption="The loss function takes as input the parameters to the model. It uses those parameters to calculate predictions based on features from the data. It compares these predictions to the labels from that data and outputs the calculated loss."
%}

## Forward Propagation Through Connected Layer
{% include chart
chart='
graph LR
    x["\(x\)"] --> wx["\(wx\)"]
    subgraph Connected Layer
    w["\(w\)"] --> wx
    wx --> wxb["\(wx+b\)"]
    b["\(b\)"] --> wxb
    end
    wxb --> y["\(y\)"]

'
caption="Forward propagation through a connected layer. The input \(x\) is multiplied by the weights \(w\) and added to the biases \(b\). The output is \(y = wx + b\)"
%}


## Backward Propagation Through Connected Layer
{% include chart
chart='
graph RL
    dLdy["\(\frac{d L}<br />{d y}\)"] --> dLdwxb["\(\frac{d L}<br />{d wx + b}\)"]
    subgraph Connected Layer
    dLdwxb --> |aggregate| dLdb["\(\frac{d L}<br />{d b}\)"]
    dLdwxb --> dLdwx["\(\frac{d L}<br />{d wx}\)"]
    dLdwx --> dLdx["\(\frac{d L}<br />{d x}\)"]
    dLdwx --> dLdw["\(\frac{d L}<br />{d w}\)"]
    w["\(w\)"] --> |"\(\frac{d wx}<br />{d x}\)"| dLdx
    end
    x["\(x\)"] --> |"\(\frac{d wx}<br />{d w}\)"|dLdw
    dLdx --> dLdxout["\(\frac{d L}<br />{d x}\)"]
'
caption="Backward propagation through a connected layer. The input is the \(\d_x\)"
%}

## Backward Propagation Through Connected Layer
{% include chart
chart='
graph RL
    dLdy["\(\d_y L\)"] --> dLdwxb["\(\d_{wx + b}L\)"]
    subgraph Connected Layer
    dLdwxb --> |aggregate| dLdb["\(\d_{b}L\)"]
    dLdwxb --> dLdwx["\(\d_{wx}L\)"]
    dLdwx --> dLdx["\(\d_{x}L\)"]
    dLdwx --> dLdw["\(\d_{w}L\)"]
    w["\(w\)"] --> |"\(\d_{x}wx = w^T\)"| dLdx
    end
    x["\(x\)"] --> |"\(\d_{w}wx = x^T\)"|dLdw
    dLdx --> dLdxout["\(\d_{x}L\)"]
'
caption="Backward propagation through a connected layer. The input is the gradient of the loss with respect to \(y\): \(\d_y L\) which is equivalent to the gradient of the loss with respect to the weighted sum: \(\d_{wx+b} L\). By aggregating over examples in the batch we get the gradient with respect to the bias: \(\d_bL\). We also know the equality \(\d_{wx + b}L = \d_{wx}L\)."
%}
