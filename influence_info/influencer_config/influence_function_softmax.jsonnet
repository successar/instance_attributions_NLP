{
    type: "influence_function_softmax",
    use_hessian: std.parseJson(std.extVar("USE_HESS")),
    normalize_grad: std.parseJson(std.extVar("NORM_GRAD"))
}