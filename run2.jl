testing_conditions = [
    "model.200.500000.0.0000001.testing",
    # "model.200.2000000.0.000001",
    # "model.200.10000000.0.0000001",
]
using DataFrames
df = DataFrame(condition = String[], dl = Float64[], star = Float64[])
for c in testing_conditions
    for rep in 1:50
        # @info "Running $(c), rep $rep"
        lpad_rep = lpad(string(rep), 2, "0")
        cmd = `python3 scripts/clade_distance.py -t1 /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees -t2 /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees.qr-dl-est`
        n1 = parse(Float32, split(read(cmd, String))[end])
        cmd = `python3 scripts/clade_distance.py -t1 /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees -t2 /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees.qr-star-est`
        n2 = parse(Float32, split(read(cmd, String))[end])
        push!(df, (c, n1, n2))
    end
end

println(describe(df))



