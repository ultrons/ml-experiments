import re

def readText(path):
    "Return a string containing cleaned up content of the file"
    return ' '.join(re.sub('[^a-z]', ' ', line.rstrip().lower()) for line in open(path))

def toInt(s):
    "'a' = 0, 'b' = 1, 'c' = 0 ..."
    return 26 if s == ' ' else ord(s[0]) - 97

def toStr(c):
    "Convert back to string"
    return ' ' if c == 26 else chr(c + 97)

def toIntSeq(s):
    return [toInt(x) for x in s]

def toStrSeq(s):
    return ''.join(toStr(x) for x in s)

def normalize(weights):
    z = sum(weights)
    return [ 1.0 * x / z for x in weights]

# HMM: (startProbs, transitionProbs, emissionProbs)
# Return mu, where mu[i][h] = P(H_i = h | E = observations)
def forwardBackward(observation, startProbs, transitionProbs, emissionProbs):
    n = len(observation)
    k = len(startProbs)
    def weights(h1, h2, i):
        # weight on the edge from (h_{i-1} = h1) to (h_{i} = h2)
        t_edge = startProbs[h1] if i == 0 else transitionProbs[h1][h2]
        e_edge = emissionProbs[h2][observation[i]]
        return t_edge * e_edge

    F={}
    B={}
    def forward(i, h):
        if i not in F:
            F[i] = {}
        if i == 0:
            F[i][h] = startProbs[h]
        else:
            if h in F[i]: return F[i][h]
            F[i][h] = sum(forward(i-1, m) * weights(m, h, i) for m in range(k))
        return F[i][h]

    def backward(i, h):
        if i not in B:
            B[i] = {}
        if i == (n - 1):
            B[i][h] = 1.
        else:
            if h in B[i]: return B[i][h]
            B[i][h] = sum(backward(i+1, m) * weights(h, m, i+1) for m in range(k))
        return B[i][h]

    mu = [None] * n
    for i in range(n):
        mu[i] = [forward(i, h) * backward(i, h) for h in range(k)]

    return mu


