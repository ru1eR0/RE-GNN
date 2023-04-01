#Here are some test commands for reproduct the corresponding results in Table 3 and 4.
python training.py --dataset Pubmed --net GCN_RW_full   --device 0 --K 2 --alpha 0.9 --lr 0.05 --weight_decay 0.0005 --dropout 0.2 --hidden 32
python training.py --dataset Pubmed --net GCNII_RW_full --device 0 --K 2 --alpha 0.1 --lr 0.05 --weight_decay 0.0005 --dropout 0   --hidden 32 --paraA 0.5 --paraB 0.1 --nlayer 8
python training.py --dataset ogbn-products --net GCN_RW_mini_G   --lr 0.01 --weight_decay 0 --dropout 0.2 --hidden 128 --alpha 0.5 --K 2 --tsplit True --batch 3 --rws 50 
python training.py --dataset pokec       --net GCNII_RW_mini_G   --lr 0.01 --weight_decay 0 --dropout 0.2 --hidden 128 --alpha 0.1 --K 2 --nlayer 8 --paraA 0.5 --paraB 0.1 --tsplit True --batch 3
