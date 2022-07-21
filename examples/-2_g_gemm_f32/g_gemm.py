import numpy as np



def g_mm(a, b, g):
    '''
    a:  m * kg
    b:  k * ng

    ret: m * ng
    '''
    M = a.shape[0]
    Ni = b.shape[1] // g
    Ki = b.shape[0]
    rets = []
    for i in range(g):
        rets.append(a[:, i*Ki : (i+1)*Ki] @ b[:, i*Ni:(i+1)*Ni])
    return np.concatenate(rets, axis=1)

def g_mm_2(a, b, g):
    '''
    a:  m * kg
    b:  kg * n

    ret: m * ng
    '''
    M = a.shape[0]
    Ki = b.shape[0] // g
    rets = []
    for i in range(g):
        rets.append(a[:, i*Ki : (i+1)*Ki] @ b[i*Ki:(i+1)*Ki:, :])
    return np.concatenate(rets, axis=1)


def get_mat_from_s(s):
    line = s.strip().split('|')
    arr = [[float(i) for i in l.strip().split()] for l in line]
    return np.array(arr, dtype=np.float32)


def get_hilbert_mat(a, b):
    return np.array(
        [[1/(i + j + 1) for j in range(b)] for i in range(a)], dtype=np.float32
    )


Mata = get_mat_from_s('''0.840188 0.394383 0.783099 0.79844 0.911647 0.197551 
| 0.335223 0.76823 0.277775 0.55397 0.477397 0.628871 
| 0.364784 0.513401 0.95223 0.916195 0.635712 0.717297 
| 0.141603 0.606969 0.0163006 0.242887 0.137232 0.804177 
| 0.156679 0.400944 0.12979 0.108809 0.998924 0.218257 
| 0.512932 0.839112 0.61264 0.296032 0.637552 0.524287 
| 0.493583 0.972775 0.292517 0.771358 0.526745 0.769914''')

Matb = get_mat_from_s('''0.400229 0.891529 0.283315 0.352458 0.807725 0.919026 0.0697553 0.949327 0.525995 0.0860558 0.192214 0.663227 
| 0.890233 0.348893 0.0641713 0.020023 0.457702 0.0630958 0.23828 0.970634 0.902208 0.85092 0.266666 0.53976''')

Matc = g_mm(Mata, Matb, 3)
#print(Matc)

G = 3
ma = get_hilbert_mat(128, 24)
mb = get_hilbert_mat(24, 300 // G)
mc = g_mm_2(ma, mb, G)

si = '\n'.join([str([str(ii)[:8] for ii in i.tolist()][0]) for i in mc])
print(si)