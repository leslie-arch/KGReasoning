#!/home/leslie/julia/1.10.0/bin/julia

#module Utils

#export flatten, format_time, eval_tuple, flatten_query;

function flatten(l)
    return collect(Iterators.flatten(l))
end

function format_time()
    return Dates.format(Dates.now(), "YY.mm.dd")
end

#---Evaluate a tuple string into a tuple.
function eval_tuple(t_string)
    #println("eval_tuple: $(t_string) - at 1:$(t_string[1])")
    #println(typeof(t_string))
    if typeof(t_string) <: Tuple
        return t_string
    end

    t_result = ()
    if !(t_string[1] in ('(', '['))
        t_result = eval(t_string)
    else
        splitted = split(t_string[2:length(t_string)-1], ",")
        List = []
        for item in splitted
            if item == ""
                continue
            end
            i = nothing
            if '.' in item
                i = tryparse(Float64, item)
            else
                i = tryparse(Int, item)
            end
            if i == nothing
                i = (item in ("none", "nothing", "null") ? nothing : item)
            end
            #item = eval(item)

            push!(List, i)
        end
        t_result = Tuple(List)
    end
    return t_result
end

function flatten_query(queries)
    all_queries = []
    for query_structure in keys(queries)
        list_queries = collect(queries[query_structure])
        append!(all_queries, [(query, query_structure) for query in list_queries])
    end
    return all_queries
end

#end # end module
