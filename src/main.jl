#!/home/leslie/julia/1.10.0/bin/julia

using MLUtils
using Random
using ArgParse
using LoggingExtras, TensorBoardLogger
using Dates
using Flux, JLD2

#println("working directory: {$(pwd())}")

include("dataloader.jl")
include("model.jl")
include("utils.jl")

using .KGDataset
using .KGModel

f_dir = "dataset";
f_model = "FB15k-betae";

query_name_dict = Dict{Tuple, String}(("e",("r",))=> "1p",
                                      ("e", ("r", "r"))=> "2p",
                                      ("e", ("r", "r", "r"))=> "3p",
                                      (("e", ("r",)), ("e", ("r",)))=> "2i",
                                      (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))=> "3i",
                                      ((("e", ("r",)), ("e", ("r",))), ("r",))=> "ip",
                                      (("e", ("r", "r")), ("e", ("r",)))=> "pi",
                                      (("e", ("r",)), ("e", ("r", "n")))=> "2in",
                                      (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))=> "3in",
                                      ((("e", ("r",)), ("e", ("r", "n"))), ("r",))=> "inp",
                                      (("e", ("r", "r")), ("e", ("r", "n")))=> "pin",
                                      (("e", ("r", "r", "n")), ("e", ("r",)))=> "pni",
                                      (("e", ("r",)), ("e", ("r",)), ("u",))=> "2u-DNF",
                                      ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))=> "up-DNF",
                                      ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",))=> "2u-DM",
                                      ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r"))=> "up-DM"
                                      );
name_query_dict = Dict{String, Tuple}((y => x) for (x, y) in query_name_dict);
all_tasks = collect(keys(name_query_dict));

function parse_cmdargs(args::Vector{String})
    s = ArgParseSettings(
        description = "Training and Testing Knowledge Graph Embedding Models",
        usage = "julia --project=[/path/to/project] src/$(@__FILE__) [<args>] [-h | --help]"
    )

    @add_arg_table s begin
        "--cuda"
        action= :store_true
        help="use GPU"
        "--train"
        action= :store_true
        help="do train"
        "--valid"
        action= :store_true
        help="do valid"
        "--test"
        action= :store_true
        help="do test"
        "--data_path"
        arg_type=String
        default= nothing
        help="KG data path"
        "-n", "--negative_sample_size"
        default=128
        arg_type=Int
        help="negative entities sampled per query"
        "-d", "--hidden_dim"
        default=500
        arg_type=Int
        help="embedding dimension"
        "-g", "--gamma"
        default=12.0
        arg_type=Float64
        help="margin in the loss"
        "-b", "--batch_size"
        default=1024
        arg_type=Int
        help="batch size of queries"
        "--test_batch_size"
        default=1
        arg_type=Int
        help="valid/test batch size"
        "--learning_rate"
        default=0.0001
        arg_type=Float64
        "--cpu"
        default=10
        arg_type=Int
        help="used to speed up torch.dataloader"
        "--save_path"
        default="."
        arg_type=String
        help="no need to set manually, will configure automatically"
        "--max_steps"
        default=100000
        arg_type=Int
        help="maximum iterations to train"
        "--warm_up_steps"
        default=nothing
        arg_type=Int
        help="no need to set manually, will configure automatically"
        "--save_checkpoint_steps"
        default=50000
        arg_type=Int
        help="save checkpoints every xx steps"
        "--valid_steps"
        default=10000
        arg_type=Int
        help="evaluate validation queries every xx steps"
        "--log_steps"
        default=100
        arg_type=Int
        help="train log every xx steps"
        "--test_log_steps"
        default=1000
        arg_type=Int
        help="valid/test log every xx steps"
        "--nentity"
        arg_type=Int
        default=0
        help="DO NOT MANUALLY SET"
        "--nrelation"
        arg_type=Int
        default=0
        help="DO NOT MANUALLY SET"
        "--geo"
        default="vec"
        arg_type=String
        help="the reasoning model, vec for GQE, box for Query2box, beta for BetaE"
        "--print_on_screen"
        action= :store_false
        "--tasks"
        default="1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"
        arg_type=String
        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task"
        "--seed"
        default=0
        arg_type=Int
        help="random seed"
        "--beta_mode"
        default="(1600,2)"
        arg_type=String
        help="(hidden_dim,num_layer) for BetaE relational projection"
        "--box_mode"
        default="(nothing,0.02)"
        arg_type=String
        help="(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist"
        "--prefix"
        default=nothing
        arg_type=String
        help="prefix of the log path"
        "--checkpoint_path"
        default=nothing
        arg_type=String
        help="path for loading the checkpoints"
        "--evaluate_union"
        default="DNF"
        arg_type=String
        help="the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\"s laws (DM)"
    end

    return parse_args(args, s)
end

#="""
Write logs to console and log file
"""=#
function set_logger(args)

    if args["train"] == true
        log_file = joinpath(args["save_path"], "train.log")
    else
        log_file = joinpath(args["save_path"], "test.log")
    end

    log_io = open(log_file, "w");
    datefmt=DateFormat("YY-mm-dd HH:MM:SS");

    timestamp_logger(logger) = TransformerLogger(logger) do log
        merge(log, (; message = "$(Dates.format(now(), datefmt)) $(log.message)"))
    end

    file_logger = timestamp_logger(FileLogger(log_file));
    global_logger(file_logger)

    if args["print_on_screen"]
        time_loger = timestamp_logger(ConsoleLogger(stdout, Logging.Info));

        tl = TeeLogger(file_logger, time_loger);
        global_logger(tl)
    end
end

#="""
Print the evaluation logs
"""=#
function log_metrics(mode, step, metrics)
    for metric in metrics
        @info "$mode $metric at step $(step): $(metrics[metric.first])"
    end
end

#="""
Evaluate queries in dataloader
"""=#
function evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer)

    average_metrics = Dict{Float}()
    all_metrics = Dict{Float}()

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics
        log_metrics(mode * " " * query_name_dict[query_structure], step, metrics[query_structure])

        for metric in metrics[query_structure]
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != "num_queries"
                average_metrics[metric] += metrics[query_structure][metric]
            end
        end
        num_queries += metrics[query_structure]["num_queries"]
        num_query_structures += 1
    end

    for metric in average_metrics
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, "average", metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    end

    log_metrics("$mode average", step, average_metrics)
    return all_metrics
end

function main(cmd_args)
    #global train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers
    args = parse_cmdargs(cmd_args)
    set_logger(args);
    Random.seed!(args["seed"])
    tasks = split(args["tasks"], ".")
    for task in tasks
        if 'n' in task && args["geo"] in ["box", "vec"]
            @assert false "Q2B and GQE cannot handle queries with negation"
        end
    end
    if args["evaluate_union"] == "DM"
        @assert args["geo"] == "beta" "only BetaE supports modeling union using De Morgan's Laws"
    end

    cur_time = format_time()
    if args["prefix"] == nothing
        prefix = "logs"
    else
        prefix = args["prefix"]
    end

    @info ("overwritting saving path: $(args["save_path"])")
    args["save_path"] = joinpath(prefix, last(split(args["data_path"], "/")), args["tasks"], args["geo"])
    geo = args["geo"]
    if geo in ["box"]
        save_str = "g-$(args["gamma"])-mode-$(args["box_mode"])"
    elseif geo in ["vec"]
        save_str = "g-$(args["gamma"])"
    elseif geo == "beta"
        save_str = "g-$(args["gamma"])-mode-$(args["beta_mode"])"
    end

    if args["checkpoint_path"] != nothing
        args["save_path"] = args["checkpoint_path"]
    else
        args["save_path"] = joinpath(args["save_path"], save_str, cur_time)
    end

    if ! ispath(args["save_path"])
        mkpath(args["save_path"])
    end

    @info ("logging to $(args["save_path"])")
    if ! args["train"] # if not training, then create tensorboard files in some tmp location
        writer = TBLogger("./logs-debug/unused-tb")
    else
        writer = TBLogger(args["save_path"])
    end
    set_logger(args)

    nentity, nrelation = open(joinpath(args["data_path"], "stats.txt")) do f
        entrel = readlines(f)
        nentity = parse(Int, last(split(entrel[1], " ")))
        nrelation = parse(Int, last(split(entrel[2], " ")))

        (nentity, nrelation)
    end

    args["nentity"] = nentity
    args["nrelation"] = nrelation

    @info(repeat("-------------------------------", 2))
    @info("Geo: $(args["geo"])")
    @info("Data Path: $(args["data_path"])")
    @info("#entity: $(nentity)")
    @info("#relation: $(nrelation)")
    @info("#max steps: $(args["max_steps"])")
    @info("Evaluate unoins using: $(args["evaluate_union"])")
    @time begin
        train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers,
        test_queries, test_hard_answers, test_easy_answers = KGDataset.load_data(args, name_query_dict)
    end
    #---------- for  test
    flatten_queries = flatten_query(train_queries)
    t = length(flatten_queries)
    println(" flatten_queries length: $(t)")
    train_data = TrainDataset(flatten_queries, train_answers, nentity, nrelation, 32)
    train_data_loader = DataLoader(train_data, batchsize = args["batch_size"], collate=true,shuffle = false);

    idx = 1
    println("train answers:")
    for a in train_answers
        println(a)
        idx += 1
        if idx > 20
            break
        end
    end
    idx = 1
    for d in train_data_loader
        println("data loader iter: xxxxxxxxxxxxxxxxxxxxx")
        println(d)
        println("---------------------------------------")
        idx += 1
        if idx > 5
            break
        end
    end

    path_data, state = iterate(train_data_loader)
    println("iterator data 1: $(path_data)\n $(state)\n")
    path_data, state = iterate(train_data_loader, state)
    println("iterator data 2: $(path_data)\n *$(state)*\n")
    println("exit after iterator.....")
    #exit()
    #

    local train_path_dataloader, train_other_dataloader
    if args["train"]
        @info("Train required...")
        train_path_queries = Dict{Any, Set}()
        train_other_queries = Dict{Any, Set}()
        query_path_list = ["1p", "2p", "3p"]
        for query_structure in keys(train_queries)
            if query_name_dict[query_structure] in query_path_list
                train_path_queries[query_structure] = train_queries[query_structure]
            else
                train_other_queries[query_structure] = train_queries[query_structure]
            end
        end

        train_path_queries = flatten_query(train_path_queries)
        @info "Flatten query length: $(length(train_path_queries)) typeof(query) $(typeof(train_path_queries))"

        train_path_dataset = KGDataset.TrainDataset(train_path_queries, train_answers, nentity, nrelation,
                                                    args["negative_sample_size"])
        train_path_dataloader = MLUtils.DataLoader(train_path_dataset, batchsize = args["batch_size"], shuffle = false);
        #train_path_iterator = KGDataset.SingleDirectionalOneShotIterator(train_path_data_loader);
        #            num_workers=args.cpu_num,
        #            collate_fn=TrainDataset.collate_fn));

        if length(train_other_queries) > 0
            train_other_queries = flatten_query(train_other_queries)
            train_other_dataset = KGDataset.TrainDataset(train_other_queries, train_answers, nentity, nrelation,
                                                         args["negative_sample_size"])
            train_other_dataloader = MLUtils.DataLoader(train_other_dataset, batchsize=args["batch_size"], shuffle=true)
            #train_other_iterator = KGDataset.SingleDirectionalOneShotIterator(train_other_dataloader)
            #_workers=args.cpu_num,
            #collate_fn=TrainDataset.collate_fn))
        else
            train_other_iterator = nothing
        end
    end

    if args["valid"]
        @info("Validation required...")

        #for query_structure in keys(valid_queries)
        #    @info query_name_dict[query_structure] * ": " * "$(length(valid_queries[query_structure]))"
        # end
        valid_queries2 = flatten_query(valid_queries)
        valid_dataloader = KGDataset.DataLoader(KGDataset.TestDataset(valid_queries2, nentity, nrelation),
                                                batchsize=args["test_batch_size"]);
        #            num_workers=args.cpu_num,
        #            collate_fn=TestDataset.collate_fn)
    end

    if args["test"]
        @info("Test requried...")

        # for query_structure in keys(test_queries)
        #    @info query_name_dict[query_structure] * ": " * "$(length(test_queries[query_structure]))"
        # end
        test_queries = flatten_query(test_queries)
        test_dataloader = KGDataset.DataLoader(
            KGDataset.TestDataset(test_queries, nentity, nrelation),
            batchsize=args["test_batch_size"]);
        #         num_workers=args.cpu_num,
        #         collate_fn=TestDataset.collate_fn)
    end

    model = KGModel.KGReasoning(nentity,
                                nrelation,
                                 args["hidden_dim"],
                                 args["gamma"],
                                 args["geo"],
                                 args["test_batch_size"],
                                 eval_tuple(args["box_mode"]),
                                 eval_tuple(args["beta_mode"]),
                                 query_name_dict,
                                 args["cuda"] == "Yes")

    @info("Model Parameter Configuration:")
    for (lindex,layer) in enumerate(Flux.params(model)) #.named_parameters()
        #@info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))
        #if param.requires_grad
        #    num_params += np.prod(param.size())
        #end
        num_params = 0
        for (pindex, pa) in enumerate(Flux.params(layer))
            @info("Parameter layer$lindex-$pindex: $(size(pa))")
            num_params += sum(length, Flux.params(layer))
        end
        @info("Parameter Number: $num_params")
    end

    if args["cuda"]
        model = model.cuda()
    end

    local init_step, checkpoint, step, current_learning_rate, warn_up_steps
    if args["train"]
        current_learning_rate = args["learning_rate"]
        opt_state = Flux.setup(Flux.Optimise.Adam(current_learning_rate), model)
        warn_up_steps = floor(args["max_steps"] / 2)
    end

    if args["checkpoint_path"] != nothing
        @info("Loading checkpoint $(args["checkpoint_path"])...")
        checkpoint = Flux.loadmodel!(model, JLD2.load(joinPath(args["checkpoint_path"], "checkpoint"), "model_state"))
        init_step = checkpoint["step"]
        Flux.loadmodel!(model, checkpoint["model_state_dict"])
        #model.load_state_dict(checkpoint["model_state_dict"])

        if args["train"]
            current_learning_rate = checkpoint["current_learning_rate"]
            warn_up_steps = checkpoint["warn_up_steps"]
            #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        end
        @info("Ramdomly Initializing $(args["geo"]) Model...")
    else
        @info("Ramdomly Initializing $(args["geo"]) Model...")
        init_step = 0
    end

    step = init_step
    if args["geo"] == "box"
        @info("box mode = $(args["box_mode"])")
    elseif args["geo"] == "beta"
        @info("beta mode = $(args["beta_mode"])")
    end
    @info("tasks = $(args["tasks"])")
    @info("init_step = $init_step")
    if args["train"]
        @info("learning_rate = $current_learning_rate")
    end
    @info("batch_size = $(args["batch_size"])")
    @info("hidden_dim = $(args["hidden_dim"])")
    @info("gamma = $(args["gamma"])")


    if args["train"]
        @info("Start Training...")
        training_logs = []
        # #Training Loop
        local path_data, path_next;
        local other_data, other_next;
        path_data, path_next = iterate(train_path_dataloader)
        other_data, other_next = iterate(train_other_dataloader)
        for step in range(init_step, args["max_steps"])
            if step == 2 * floor(args["max_steps"] / 3)
                args["valid_steps"] *= 4
            end


            println("path_data: $(path_data),\n path_next: $(path_next)")
            log = KGModel.train_step(model, opt_state, path_data, args, step)
            # data for next step
            path_data, path_next = iterate(train_path_dataloader, path_next)
            for metric in log
                writer.add_scalar("path_" * metric, log[metric], step)
            end

            if train_other_iterator != nothing
                log = KGModel.train_step(model, opt_state, train_other_dataloader, args, step)
                other_data, other_next = iterate(train_other_dataloader, other_next)
                for metric in log
                    @info "metric : $(metric)"
                    writer.add_scalar("other_"+metric, log[metric], step)
                end
                log = KGModel.train_step(model, opt_state, train_path_dataloader, args, step)
                other_data, other_next = iterate(train_other_dataloader, other_next)
            end

            training_logs.append(log)

            if step >= warn_up_steps
                current_learning_rate = current_learning_rate / 5
                @info("Change learning_rate to $(current_learning_rate) at step $(step)")

                opt_state = Flux.setup(Flux.Optimiser.Adam(lr = current_learning_rate),
                                       model)
                warn_up_steps = warn_up_steps * 1.5
            end

            if step % args.save_checkpoint_steps == 0
                save_variable_list = (
                    "step": step,
                    "current_learning_rate": current_learning_rate,
                    "warm_up_steps": warn_up_steps
                )
                JLD2.save(model, opt_state, save_variable_list, args)
            end

            if step % args["valid_steps"] == 0 && step > 0
                if args["do_valid"]
                    @info("Evaluating on Valid Dataset...")
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args,
                                                 valid_dataloader, query_name_dict, "Valid", step, writer)
                end

                if args["do_test"]
                    @info("Evaluating on Test Dataset...")
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args,
                                                test_dataloader, query_name_dict, "Test", step, writer)
                end
            end

            if step % args["log_steps"] == 0
                metrics = Dict()
                for metric in training_logs[0].keys()
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                end

                log_metrics("Training average", step, metrics)
                training_logs = []
            end

            save_variable_list = (
                "step": step,
                "current_learning_rate": current_learning_rate,
                "warm_up_steps": warn_up_steps
            )
            JLD2.save(model, opt_state, save_variable_list, args)

            try
                print(step)
            catch
                step = 0
            end
        end
        @info("Training finished!!")
    end

    #    if args["test"]
    #        @info("Evaluating on Test Dataset...")
    #        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, "Test", step, writer)
    #    end
    #
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = Vector{String}(["--train", "--data_path", "dataset/FB15k-betae",
                           "-n", "32", "-b", "4", "-d", "800", "-g", "24","--learning_rate",
                           "0.0001", "--max_steps", "450001",
                           "--cpu", "1", "--geo", "beta", "--valid_steps", "15000"])
    #structed_args = parse_cmdargs(args);
    #println(structed_args)

    main(args)

end
