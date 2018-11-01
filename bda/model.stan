data {
    int <lower=0> n[4]; 
    int <lower=0> y[4];
    vector[4] x;
    vector[2] mean;
    matrix[2, 2] cov;
}
parameters {
    vector[2] theta
}
model {
    pr ~ multi_normal(mean, cov)
    y ~ binomial_logit(n, theta[0] + theta[1]*x);
}