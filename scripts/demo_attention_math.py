import numpy as np

# 1 query vector (imagine: "the LUT query asking about warmth")
query = np.array([1.0, 0.0, 1.0, 0.0])

# 3 key/value pairs (imagine: 3 scene tokens -- e.g. sky, skin, foliage)
keys = np.array([
    [1.0, 0.0, 1.0, 0.0],   # key 0 -- very similar direction to the query
    [0.0, 1.0, 0.0, 1.0],   # key 1 -- quite different from the query
    [0.5, 0.5, 0.5, 0.5],   # key 2 -- somewhere in between
])
values = np.array([
    [10.0, 10.0, 10.0, 10.0],   # value 0
    [0.0,   0.0,  0.0,  0.0],   # value 1
    [5.0,   5.0,  5.0,  5.0],   # value 2
])

d = query.shape[0]  # dimension of each vector, here 4

# step 1: raw similarity score between the query and EACH key (plain dot product)
scores = keys @ query   # shape [3] -- one score per key
print("raw scores:", scores)

# step 2: scale down by sqrt(dimension) -- keeps scores from growing huge as
# dimension increases, which would otherwise make softmax too extreme/unstable
scaled_scores = scores / np.sqrt(d)
print("scaled scores:", scaled_scores)

# step 3: softmax turns scores into weights that are all positive and sum to 1
def softmax(x):
    e = np.exp(x - x.max())   # subtract max first, purely for numerical stability
    return e / e.sum()

weights = softmax(scaled_scores)
print("attention weights:", weights, " (sums to:", weights.sum(), ")")

# step 4: the output is a weighted blend of the VALUES, using those weights
output = weights @ values   # shape [4]
print("output:", output)