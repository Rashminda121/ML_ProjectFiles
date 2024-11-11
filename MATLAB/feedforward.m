% training using feedforward
clc,clearvars;

n = 100;

x1 = rand(n, 10);
x2 = rand(n, 10);
x3 = rand(n, 10);
x4 = rand(n, 10);

% Y = 10*sin(x1)+sqrt(x2)+5*x3+0.05*x4.^0.6-.003*x5.^3-log (x6) + 2*randn (N,1);

% Layer 1: Initial transformations
layer1_x1 = 2 * x1;
layer1_x2 = log(x2 + 1);
layer1_x3 = sqrt(x3);
layer1_x4 = x4.^2;

% Layer 2: Combine transformed features
layer2 = layer1_x1 + 3 * layer1_x2 - 0.5 * layer1_x3 + 4 * layer1_x4;

% Layer 3: Non-linear combination
layer3 = tanh(layer2);

% Final output y with added noise
y = layer3 + 0.1 * randn(n, 10);

