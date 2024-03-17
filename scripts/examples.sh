
# FB15k-237
CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo vec --valid_steps 15000 \
                       --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo beta --valid_steps 15000 \
                       --beta_mode "(1600,2)"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo box --valid_steps 15000 \
                       --box_mode "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


# FB15k
CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo vec --valid_steps 15000 \
                       --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-betae -n 128 -b 512 -d 400 -g 60 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo beta --valid_steps 15000 \
                       --beta_mode "(1600,2)"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/FB15k-betae -n 128 -b 512 -d 400 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo box --valid_steps 15000 \
                       --box_mode "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


# NELL
CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/NELL-betae -n 128 -b 512 -d 800 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo vec --valid_steps 15000 \
                       --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/NELL-betae -n 128 -b 512 -d 400 -g 60 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo beta --valid_steps 15000 \
                       --beta_mode "(1600,2)"

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --train --valid --test \
                       --data_path dataset/NELL-betae -n 128 -b 512 -d 400 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo box --valid_steps 15000 \
                       --box_mode "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


## Evaluation

CUDA_VISIBLE_DEVICES=0 julia --project=. src/main.jl --cuda --test \
                       --data_path dataset/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
                       --learning_rate 0.0001 --max_steps 450001 --cpu 1 --geo box --valid_steps 15000 \
                       --box_mode "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --checkpoint_path $CKPT_PATH
