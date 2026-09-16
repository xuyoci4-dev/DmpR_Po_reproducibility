import numpy as np
BASES='ACGT'
def encode_880(sequences):
    sequences=list(sequences); out=np.zeros((len(sequences),880),np.float32); mapping={b:i for i,b in enumerate(BASES)}
    for row,sequence in enumerate(sequences):
        if len(sequence)!=156 or set(sequence)-set(BASES): raise ValueError(f'Expected A/C/G/T 156-bp sequence at row {row}')
        for position,base in enumerate(sequence): out[row,position*4+mapping[base]]=1
        for position in range(153):
            value=0
            for base in sequence[position:position+4]: value=value*4+mapping[base]
            out[row,624+value]+=1/153
    return out
def encode_cnn(sequences): return encode_880(sequences)[:,:624].reshape(-1,156,4).transpose(0,2,1).astype(np.float32)
