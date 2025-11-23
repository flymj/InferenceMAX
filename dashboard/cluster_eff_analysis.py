"""简化的聚类效率分析"""
import sys,os
sys.path.append(os.getcwd())
from dashboard.operators.flash_attention_operator import FlashAttentionOperator
from hardware_descriptions import FlashAttentionHardware
from collections import defaultdict

h100 = FlashAttentionHardware(tc_tflops=989.0,fp32_tflops=60.0,hbm_tbs=3.35,freq_ghz=1.83,num_sms=132,name="H100")

def parse(line):
    parts=line.strip().split('\t')
    if len(parts)!=2:return None,None
    cfg_str,cycles_str=parts
    actual=int(cycles_str)
    config={}
    for p in cfg_str.split(','):
        if ':' in p:
            k,v=p.split(':',1)
            config[k]=v
    w={"batch":int(config.get("batch_size",1)),"heads":int(config.get("num_heads",1)),
       "kv_heads":int(config.get("num_heads_k",config.get("num_heads",1))),"d":int(config.get("head_dim",128)),
       "nq":int(config.get("seqlen_q",512)),"nk":int(config.get("seqlen_k",512)),
       "mask_type":"causal" if config.get("custom_mask")=="1" else "dense","dropout":float(config.get("dropout",0.0)),
       "is_flash_v2":True,"fixed_overhead_us":0.0,"compute_efficiency":1.0}
    return w,actual

def calc_eff(w,actual):
    op=FlashAttentionOperator(w)
    import pdb; pdb.set_trace()
    ti=op.calculate_tflops(h100)
    hi=op.calculate_hbm_throughput(h100)
    pred=max(ti["t_tensor"]*1e6,hi["t_hbm"]*1e6)*h100.freq_ghz*1e3
    return pred/actual if actual>0 else 0

if __name__=="__main__":
    fname=sys.argv[1] if len(sys.argv)>1 else "all_user_cases.txt"
    cases=[]
    with open(fname,'r') as f:
        for line in f:
            w,act=parse(line)
            if not w or not act:continue
            eff=calc_eff(w,act)
            seq=w["nq"]
            batch=w["batch"]
            gqa_ratio=w["heads"]/w["kv_heads"]
            cases.append((seq,batch,gqa_ratio,w["heads"],w["d"],w["mask_type"],w["dropout"],eff,act))
    
    print(f"Total: {len(cases)} cases\n")
    
    # By Seq
    groups=defaultdict(list)
    for seq,*_,eff,_ in cases:
        if seq<512:cat="Tiny(<512)"
        elif seq<1024:cat="Small(512-1K)"
        elif seq<2048:cat="Med(1K-2K)"
        elif seq<8192:cat="Large(2K-8K)"
        else:cat="Huge(>8K)"
        groups[cat].append(eff)
    
    print("BY SEQUENCE LENGTH:")
    print(f"{'Category':<20} {'Count':<10} {'Avg Eff':<12}")
    print("-"*50)
    for cat in ["Tiny(<512)","Small(512-1K)","Med(1K-2K)","Large(2K-8K)","Huge(>8K)"]:
        if cat in groups:
            effs=groups[cat]
            print(f"{cat:<20} {len(effs):<10} {sum(effs)/len(effs):.3f}")
    
    # By GQA
    groups=defaultdict(list)
    for _,_,gqa,*rest,eff,_ in cases:
        if gqa==1:cat="MHA(1:1)"
        elif gqa<=2:cat="GQA-Lite(2:1)"
        elif gqa<=4:cat="GQA-Med(3-4:1)"
        else:cat="GQA-Heavy(>4:1)"
        groups[cat].append(eff)
    
    print("\nBY GQA RATIO:")
    print(f"{'Category':<20} {'Count':<10} {'Avg Eff':<12}")
    print("-"*50)
    for cat in ["MHA(1:1)","GQA-Lite(2:1)","GQA-Med(3-4:1)","GQA-Heavy(>4:1)"]:
        if cat in groups:
            effs=groups[cat]
            print(f"{cat:<20} {len(effs):<10} {sum(effs)/len(effs):.3f}")
    
    # By Mask
    groups=defaultdict(list)
    for *_,mask,drop,eff,_ in cases:
        groups[mask].append(eff)
    
    print("\nBY MASK TYPE:")
    print(f"{'Category':<20} {'Count':<10} {'Avg Eff':<12}")
    print("-"*50)
    for cat in ["causal","dense"]:
        if cat in groups:
            effs=groups[cat]
            print(f"{cat:<20} {len(effs):<10} {sum(effs)/len(effs):.3f}")
