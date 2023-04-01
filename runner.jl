# using Threads
testing_conditions = [
    "model.200.500000.0.0000001.testing",
    # "model.200.2000000.0.000001",
    # "model.200.10000000.0.0000001",
]

for rep in 1:50
    for c in testing_conditions
        @info "Running $(c), rep $rep"
        lpad_rep = lpad(string(rep), 2, "0")
        cmd1 = `python3 quintet_rooting.py -t /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees -g /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/estimatedgenetre -c STAR -abratio 0 -coef 0.01 -o /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees.qr-star-est`
        # lpad_rep = lpad(string(rep), 2, "0")
        cmd2 = `python3 quintet_rooting.py -t /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees -g /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/estimatedgenetre -c dl -o /Users/lbq/research/contrastive-qr/data/$(c)/$(lpad_rep)/s_tree.trees.qr-dl-est`
        begin
            s1 = Threads.@spawn run(cmd1)
            s2 = Threads.@spawn run(cmd2)
            s1 = fetch(s1)
            s2 = fetch(s2)
        end
    end
end

for rep in 1:50
    Threads.@threads for c in testing_conditions
        
    end
end

