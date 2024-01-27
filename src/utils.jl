#!/home/leslie/julia/1.10.0/bin/julia

function format_time()
    return Dates.format(Dates.now(), "YY.mm.dd")
end

#---Evaluate a tuple string into a tuple.
function eval_tuple(arg_return)
    if typeof(arg_return) <: Tuple
        return arg_return
    end

    if !(arg_return[1] in ("(", "["))
        arg_return = eval(arg_return)
    else
        splitted = split(arg_return[2:length(arg_return)-1], ", ")
        List = []
        for item in splitted
            try
                item = eval(item)
            catch err
                pass
            end
            if item == ""
                continue
            end
            append!(List, item)
        end
        arg_return = tuple(List)
        return arg_return
    end
end

function flatten_query(queries)
    all_queries = []
    for query_structure in keys(queries)
        list_queries = collect(queries[query_structure])
        ttt = [(query, query_structure) for query in list_queries]
        #println("query_structure key: $(query_structure) queries length: $(length(list_queries)) tuple length: $(length(ttt))");
        append!(all_queries, [(query, query_structure) for query in list_queries])
    end
    return all_queries
end
