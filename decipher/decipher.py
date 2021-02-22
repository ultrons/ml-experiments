import util

# Size of state space
K = 27
corpus = 'tolstoy.txt'
cipher_text = 'coded_message.txt'
num_itr = 200

### Initialize the parameters

# Uniform
startProbs = [1.0 / K for _ in range(K)]

# Transition Prob (estimated using maximum likelihood)
chars = util.toIntSeq(util.readText(corpus))
transCounts = [[0 for h2 in range(K)] for h1 in range(K)]
for i in range(1, len(chars)):
    h1, h2 = chars[i - 1], chars[i]
    transCounts[h1][h2] += 1

transitionProbs = [None] * K
for h1 in range(K):
    transitionProbs[h1] = util.normalize(transCounts[h1])

observations = util.toIntSeq(util.readText(cipher_text))
# Emission
emissionProbs = [[1.0 / K for h2 in range(K)] for h1 in range(K)]

for _ in range(num_itr):
    # E-Step
    mu = util.forwardBackward(observations, startProbs, transitionProbs, emissionProbs)

    print(util.toStrSeq([max((p, h) for h, p in enumerate(prob))[1] for prob in mu]))
    print ""

    # M-Step
    emissionCounts = [[0 for e in range(K)] for h in range(K)]
    for i, probs in enumerate(mu):
        e = observations[i]
        for h in range(K):
            emissionCounts[h][e] += probs[h]
    for h in range(K):
        emissionProbs[h] = util.normalize(emissionCounts[h])
